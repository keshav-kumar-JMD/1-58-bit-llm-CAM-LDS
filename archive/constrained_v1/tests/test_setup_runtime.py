import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camlds.setup_runtime import create_worker_environment


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_avoids_ensurepip_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory)
            marker = environment / 'existing-package'
            marker.write_text('keep me')
            executable = environment / 'bin/python'
            with patch('camlds.setup_runtime.venv.EnvBuilder') as builder, \
                 patch('camlds.setup_runtime.subprocess.run', return_value=types.SimpleNamespace(returncode=0)) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                create_worker_environment(environment, executable)
            builder.assert_called_once_with(with_pip=False, system_site_packages=True)
            builder.return_value.create.assert_called_once_with(environment)
            self.assertEqual(run.call_args_list[0].args[0], [sys.executable, '-m', 'pip', '--python', str(executable), 'install', '--ignore-installed', 'pip'])
            self.assertEqual(run.call_args_list[1].args[0][:2], [str(executable), '-c'])
            self.assertEqual(marker.read_text(), 'keep me')

    def test_bootstrap_failure_retains_log_and_does_not_run_import_check(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory)
            def fail(command, **kwargs):
                kwargs['stdout'].write('Specific pip failure detail\n')
                return types.SimpleNamespace(returncode=1)
            with patch('camlds.setup_runtime.venv.EnvBuilder'), \
                 patch('camlds.setup_runtime.subprocess.run', side_effect=fail) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Specific pip failure detail'):
                    create_worker_environment(environment, environment/'bin/python')
            self.assertEqual(run.call_count, 1)
            self.assertIn('Specific pip failure detail', (environment/'bootstrap.log').read_text())


if __name__ == '__main__':
    unittest.main()
