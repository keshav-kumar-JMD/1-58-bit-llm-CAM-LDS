import contextlib
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from camlds.sync import Mirror, CheckpointTransferError


def rejected(code, body):
    return HTTPError('https://receiver.example/row/qwen1b', code, 'Rejected', {},
                     io.BytesIO(json.dumps(body).encode()))


class TransferTests(unittest.TestCase):
    def test_transient_legacy_409_retries_identical_saved_response(self):
        mirror = Mirror('https://receiver.example', 'secret-token')
        row = {'prompt id':'252', 'raw response':'ATT ATT { unfinished', 'status':'completed'}
        with patch('camlds.sync.urlopen', side_effect=[
            rejected(409, {'error':'Permission denied'}), io.BytesIO(b'{"saved":true}')]) as request, \
                patch('camlds.sync.time.sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
            mirror.row('qwen1b', row)
        self.assertEqual(request.call_count, 2)
        self.assertEqual([json.loads(c.args[0].data) for c in request.call_args_list], [row, row])
        sleep.assert_called_once_with(2)

    def test_network_and_retryable_file_lock_recover(self):
        with patch('camlds.sync.urlopen', side_effect=[URLError('connection reset'),
            rejected(503, {'error':'PermissionError: file is locked', 'retryable':True}),
            io.BytesIO(b'{"saved":true}')]), patch('camlds.sync.time.sleep') as sleep, \
                contextlib.redirect_stdout(io.StringIO()):
            result = Mirror('https://receiver.example','token').call('/row/qwen1b', {})
        self.assertTrue(result['saved'])
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2,5])

    def test_real_conflict_and_bad_token_do_not_retry(self):
        for code, body in [(409, {'error':'Input fields changed', 'retryable':False}),
                           (401, {'error':'Unauthorized'})]:
            with patch('camlds.sync.urlopen', side_effect=rejected(code, body)) as request, \
                    patch('camlds.sync.time.sleep') as sleep:
                with self.assertRaisesRegex(CheckpointTransferError, body['error']):
                    Mirror('https://receiver.example','token').call('/row/qwen1b', {})
            self.assertEqual(request.call_count, 1)
            sleep.assert_not_called()

    def test_persistent_outage_is_bounded_and_redacts_token(self):
        with patch('camlds.sync.urlopen', side_effect=TimeoutError('secret-token timeout')) as request, \
                patch('camlds.sync.time.sleep') as sleep, contextlib.redirect_stdout(io.StringIO()) as log:
            with self.assertRaises(CheckpointTransferError) as failure:
                Mirror('https://receiver.example','secret-token').call('/row/qwen1b', {})
        self.assertEqual(request.call_count, 5)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2,5,10,20])
        self.assertIn('checkpoints are preserved', str(failure.exception))
        self.assertNotIn('secret-token', str(failure.exception) + log.getvalue())
