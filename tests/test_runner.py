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
from camlds.core import Results, digest, load_queries, OUTPUT_SUBDIR
from camlds.runner import worker, summary, run_suite, print_report
from camlds.sync import Mirror, CheckpointTransferError
from test_core import record, valid


class FakeBackend:
    context = 4096
    profile = {'test': True}
    count = staticmethod(lambda text: len(text)//4)
    def generate(self, *args):
        return dict(raw='unstructured response', input_tokens=100, output_tokens=100, finish='stop')
    def close(self):
        pass


class RunnerTests(unittest.TestCase):
    def test_transfer_failure_preserves_response_and_resume_skips_inference(self):
        from unittest.mock import Mock
        rows = [record()]
        api = types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha='pinned')))
        mirror = types.SimpleNamespace(restore=Mock(), initialize=Mock(),
            row=Mock(side_effect=CheckpointTransferError('receiver unavailable')))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'huggingface_hub':api}), \
                patch('camlds.runner.load_queries',return_value=rows), \
                patch('camlds.runner.gpu_info',return_value={}), \
                patch('camlds.sync.Mirror.from_env',return_value=mirror):
            with patch('camlds.runner.load_backend',return_value=FakeBackend()):
                with self.assertRaises(CheckpointTransferError):
                    worker(directory,'qwen1b')
            self.assertEqual(summary(directory)[0]['completed'],1)
            mirror.row.side_effect = None
            with patch('camlds.runner.load_backend') as load:
                worker(directory,'qwen1b')
                load.assert_not_called()
            # Resume sends the already collected response in its initialization.
            saved = mirror.initialize.call_args.args[2]
            self.assertEqual(saved.rows[1]['raw response'],'unstructured response')

    def test_gguf_generation_has_no_output_constraints(self):
        from unittest.mock import Mock
        from camlds.models import GGUFBackend
        backend = GGUFBackend.__new__(GGUFBackend)
        backend.context = 4096
        backend.formatted = lambda prompt: ([1,2,3], ['<|im_end|>'])
        backend.model = types.SimpleNamespace(reset=Mock(), create_completion=Mock(return_value={
            'choices':[{'text':'ATT ATT {', 'finish_reason':'length'}],
            'usage':{'prompt_tokens':3,'completion_tokens':2048}}))
        self.assertEqual(backend.generate('original prompt',2048)['raw'],'ATT ATT {')
        settings = backend.model.create_completion.call_args.kwargs
        self.assertNotIn('grammar',settings)
        self.assertNotIn('logits_processor',settings)
        self.assertEqual(settings['repeat_penalty'],1.0)
        self.assertEqual(settings['temperature'],0.0)

    def test_native_crash_does_not_block_later_rows(self):
        rows = [record(str(i)) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/OUTPUT_SUBDIR
            output.mkdir(parents=True)
            path = output/'llm_responses_qwen1b_camlds.csv'
            Results(path,rows,'test')
            (output/'qwen1b.manifest.json').write_text(json.dumps({'config_hash':'test'}))
            launches = []
            def launch(command, **kwargs):
                requested = set(map(int,command[command.index('--ids')+1].split(',')))
                launches.append(requested)
                saved = Results(path,rows,'test')
                if len(launches) == 1:
                    saved.begin(1)
                    return types.SimpleNamespace(pid=10,poll=lambda:-11)
                for index in requested:
                    saved.begin(index)
                    saved.update(index,'unstructured response')
                return types.SimpleNamespace(pid=11,poll=lambda:0)
            with patch('camlds.runner.load_queries',return_value=rows), patch('camlds.runner.gpu_info',return_value={'free_gib':40}), patch('camlds.runner.subprocess.Popen',side_effect=launch), contextlib.redirect_stdout(io.StringIO()):
                report = run_suite(directory,selected=['qwen1b'],ids=[1,2,3],passes=1)
            self.assertEqual(launches,[{1,2,3},{2,3}])
            self.assertEqual(report['coverage'][0]['completed'],2)
            saved = Results(path,rows,'test')
            self.assertEqual(saved.rows[1]['status'],'execution_error')
            self.assertEqual(saved.rows[2]['raw response'],'unstructured response')

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

    def test_report_counts_only_requested_rows(self):
        report = {'coverage':[{'file':'bitnet.csv', 'retryable_ids':[i for i in range(1,991) if i not in [2,89,172]]}], 'incomplete':{}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            print_report(report, ids=[2,89,172])
        self.assertIn('3/3 responses collected; 0 remaining', output.getvalue())
        self.assertLess(len(output.getvalue()), 150)

    def test_supervisor_stops_on_checkpoint_io_failure(self):
        process = types.SimpleNamespace(pid=1234, poll=lambda: 73)
        with tempfile.TemporaryDirectory() as directory, patch('camlds.runner.gpu_info', return_value={'free_gib':80}), patch('camlds.runner.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'checkpoint I/O failed'):
                run_suite(directory, selected=['qwen1b'])

    def test_worker_continues_after_error_and_never_repeats_collected_text(self):
        rows = [record(str(i)) for i in range(3)]
        api = types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(model_info=lambda repo: types.SimpleNamespace(sha='pinned-revision')))
        class Backend(FakeBackend):
            calls = []
            def generate(self, prompt, cap):
                self.calls.append(prompt)
                if prompt.endswith('0'):
                    raise ValueError('simulated per-row execution error')
                return dict(raw='',input_tokens=10,output_tokens=0,finish='stop')
        backend = Backend()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'huggingface_hub':api}), patch('camlds.runner.load_queries', return_value=rows), patch('camlds.runner.gpu_info', return_value={}), patch('camlds.sync.Mirror.from_env', return_value=None):
                with patch('camlds.runner.load_backend',return_value=backend):
                    worker(directory,'qwen1b')
                self.assertEqual(len(backend.calls),3)
                self.assertEqual(summary(directory)[0]['completed'],2)
                with patch('camlds.runner.load_backend',return_value=FakeBackend()):
                    worker(directory,'qwen1b')
                self.assertEqual(summary(directory)[0]['completed'],3)
                with patch('camlds.runner.load_backend') as load:
                    worker(directory,'qwen1b')
                    load.assert_not_called()
                manifest=json.loads((Path(directory)/OUTPUT_SUBDIR/'qwen1b.manifest.json').read_text())
                self.assertEqual(manifest['policy']['mode'],'raw-baseline')
                self.assertEqual(manifest['revision'],'pinned-revision')

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
                with self.assertRaisesRegex(CheckpointTransferError, 'HTTP 401'):
                    Mirror(f'http://127.0.0.1:{port}', 'invalid').call('/snapshot/qwen1b')
                self.assertEqual(mirror.call('/health/qwen1b')['protocol'], 'raw-baseline-v2')
                self.assertEqual(mirror.call('/snapshot/qwen1b'), {})
                rows = load_queries(root/'data/llm_queries.json')
                manifest = dict(model='qwen1b', dataset_hash=digest([{k:v for k,v in r.items() if k != '_recovery'} for r in rows]), config_hash='test')
                client = root/'client'
                client.mkdir()
                results = Results(client/'llm_responses_qwen1b_camlds.csv', rows, 'test')
                mirror.initialize('qwen1b', manifest, results)
                results.update(89, 'raw malformed { response', attempts=1)
                with self.assertRaisesRegex(CheckpointTransferError, 'Input fields changed'):
                    mirror.row('qwen1b', {**results.rows[89], 'prompt':'changed'})
                mirror.row('qwen1b', results.rows[89])
                restored = root/'restored'
                restored.mkdir()
                mirror.restore('qwen1b', restored)
                recovered = Results(restored/'llm_responses_qwen1b_camlds.csv', rows, 'test')
                self.assertEqual(recovered.rows[89]['status'], 'completed')
                self.assertEqual(len(recovered.pending()), 989)
                # Reinitializing with an older snapshot must not erase a success.
                stale = Results(root/'stale.csv', rows, 'test')
                mirror.initialize('qwen1b', manifest, stale)
                self.assertEqual(stale.rows[89]['raw response'], 'raw malformed { response')
                self.assertEqual(Results(root/OUTPUT_SUBDIR/'llm_responses_qwen1b_camlds.csv', rows, 'test').rows[89]['status'], 'completed')
            finally:
                process.terminate()
                process.wait(timeout=10)
                process.stdout.close()
                process.stderr.close()


if __name__ == '__main__':
    unittest.main()
