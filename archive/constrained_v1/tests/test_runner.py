import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'.cache/audit-packages'))
from camlds.core import Results, digest, load_queries
from camlds.runner import worker, summary, run_suite, print_report
from camlds.sync import Mirror
from test_core import record, valid


class FakeBackend:
    context = 4096
    profile = {'test': True}
    count = staticmethod(lambda text: len(text)//4)
    def generate(self, *args):
        return dict(raw=json.dumps(valid()), input_tokens=100, output_tokens=100, finish='stop')
    def close(self):
        pass


class RunnerTests(unittest.TestCase):
    def test_cuda_graph_workaround_and_native_crash_reporting(self):
        process = types.SimpleNamespace(pid=1234, poll=lambda: -11)
        with tempfile.TemporaryDirectory() as directory, patch('camlds.runner.gpu_info', return_value={'free_gib':40}), patch('camlds.runner.subprocess.Popen', return_value=process) as launch, contextlib.redirect_stdout(io.StringIO()) as output:
            report = run_suite(directory, selected=['qwen1b'], ids=[2,89,172], passes=1)
        env = launch.call_args.kwargs['env']
        self.assertEqual(env['GGML_CUDA_DISABLE_GRAPHS'], '1')
        self.assertEqual(env['PYTHONFAULTHANDLER'], '1')
        self.assertIn('SIGSEGV', output.getvalue())
        self.assertIn('qwen1b', report['incomplete'])

    def test_low_memory_retry_disables_flash_for_every_profile(self):
        from camlds.models import gguf_profiles
        profiles = gguf_profiles(True, 40, low_memory=True)
        self.assertEqual(profiles[0], (-1,128,32,False))
        self.assertTrue(all(not profile[3] for profile in profiles))

    def test_smoke_report_counts_only_requested_rows(self):
        report = {'coverage':[{'file':'bitnet.csv', 'retryable_ids':[i for i in range(1,991) if i not in [2,89,172]]}], 'incomplete':{}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            print_report(report, ids=[2,89,172])
        self.assertIn('3/3 requested rows valid; 0 remaining', output.getvalue())
        self.assertLess(len(output.getvalue()), 150)

    def test_supervisor_stops_on_checkpoint_io_failure(self):
        process = types.SimpleNamespace(pid=1234, poll=lambda: 73)
        with tempfile.TemporaryDirectory() as directory, patch('camlds.runner.gpu_info', return_value={'free_gib':80}), patch('camlds.runner.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'checkpoint I/O failed'):
                run_suite(directory, selected=['qwen1b'])

    def test_worker_resumes_gaps_and_records_engine_error(self):
        rows = [record(str(i)) for i in range(3)]
        api = types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha='pinned-revision')))
        class FailingBackend(FakeBackend):
            def generate(self, *args):
                raise RuntimeError('simulated CUDA OOM')
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'huggingface_hub':api}), patch('camlds.runner.load_queries', return_value=rows), patch('camlds.runner.gpu_info', return_value={}), patch('camlds.sync.Mirror.from_env', return_value=None):
                with patch('camlds.runner.load_backend', return_value=FailingBackend()):
                    with self.assertRaisesRegex(RuntimeError, 'OOM'):
                        worker(directory, 'qwen1b', ids={2})
                with patch('camlds.runner.load_backend', return_value=FakeBackend()):
                    worker(directory, 'qwen1b', ids={3})
                    self.assertEqual(summary(directory)[0]['retryable_ids'], [1,2])
                    worker(directory, 'qwen1b')
                    self.assertEqual(summary(directory)[0]['valid'], 3)
                    worker(directory, 'qwen1b')  # complete run does not load model
                manifest = json.loads((Path(directory)/'results/qwen1b.manifest.json').read_text())
                self.assertEqual(manifest['revision'], 'pinned-revision')

    def test_receiver_mirror_roundtrip_and_auth(self):
        from urllib.error import HTTPError
        import shutil
        import socket
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').mkdir()
            shutil.copyfile(ROOT/'data/llm_queries.json', root/'data/llm_queries.json')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = {**os.environ, 'PYTHONPATH':str(ROOT)}
            process = subprocess.Popen([sys.executable, '-u', '-m', 'camlds.local_store', '--project', str(root), '--port', str(port)],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                process.stdout.readline()
                token_line = process.stdout.readline()
                token = token_line.split(': ', 1)[1].strip()
                mirror = Mirror(f'http://127.0.0.1:{port}', token)
                with self.assertRaises(HTTPError) as captured:
                    Mirror(f'http://127.0.0.1:{port}', 'invalid').call('/snapshot/qwen1b')
                captured.exception.close()
                self.assertEqual(mirror.call('/snapshot/qwen1b'), {})
                rows = load_queries(root/'data/llm_queries.json')
                manifest = dict(model='qwen1b', dataset_hash=digest([{k:v for k,v in r.items() if k != '_recovery'} for r in rows]), config_hash='test')
                client = root/'client'
                client.mkdir()
                results = Results(client/'llm_responses_qwen1b_camlds.csv', rows, 'test')
                mirror.initialize('qwen1b', manifest, results)
                results.update(89, valid(), attempts=1)
                mirror.row('qwen1b', results.rows[89])
                restored = root/'restored'
                restored.mkdir()
                mirror.restore('qwen1b', restored)
                recovered = Results(restored/'llm_responses_qwen1b_camlds.csv', rows, 'test')
                self.assertEqual(recovered.rows[89]['status'], 'ok')
                self.assertEqual(len(recovered.pending()), 989)
                # Reinitializing with an older snapshot must not erase a success.
                stale = Results(root/'stale.csv', rows, 'test')
                mirror.initialize('qwen1b', manifest, stale)
                self.assertEqual(Results(root/'results/llm_responses_qwen1b_camlds.csv', rows, 'test').rows[89]['status'], 'ok')
            finally:
                process.terminate()
                process.wait(timeout=10)
                process.stdout.close()
                process.stderr.close()


if __name__ == '__main__':
    unittest.main()
