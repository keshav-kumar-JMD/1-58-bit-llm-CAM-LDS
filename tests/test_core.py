import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.cache' / 'audit-packages'))
from camlds.core import *
from camlds.runner import evaluate


def valid():
    return dict(techniques=[f'T{i:04}' for i in range(1000, 1010)], confidence='Neutral', explanation='Observed process activity.')


def record(log='log line'):
    return dict(scenario='1', variant='x', step='1', prompt=PROMPT_START+'\n\n'+log,
                ground_truth=['T1001'], _recovery='')


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def read_text(self, text, expected=1):
        path = self.directory / 'input.json'
        path.write_text(text, encoding='utf-8')
        return load_queries(path, expected)

    def test_real_990_unchanged(self):
        rows = load_queries(ROOT/'data/llm_queries.json')
        original = [json.loads(line) for line in (ROOT/'data/llm_queries.json').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        self.assertEqual(len(rows), 990)
        for a,b in zip(rows, original):
            self.assertEqual({k:a[k] for k in INPUT_KEYS}, b)

    def test_array_jsonl_and_blanks(self):
        r = record('unclosed nested log: {"field":')
        for text in [json.dumps([r]), json.dumps(r)+'\n\n', '\ufeff'+json.dumps(r)]:
            self.assertEqual(self.read_text(text)[0]['prompt'], r['prompt'])

    def test_recover_truncated_outer_json_and_literal_newline(self):
        r = record('first\n"quoted" Unicode café')
        raw = json.dumps(r, ensure_ascii=False)
        # Truncated end of outer object; complete prompt and truth remain.
        row = self.read_text(raw[:-1])[0]
        self.assertEqual(row['prompt'], r['prompt'])
        self.assertTrue(row['_recovery'])
        raw = raw.replace('first\\n', 'first\n')
        self.assertEqual(self.read_text(raw)[0]['prompt'], r['prompt'])

    def test_recover_missing_metadata_truth_spelling(self):
        text = '{"prompt": "'+PROMPT_START+'\\n\\nlog", "ground truth": ["T1000"]'
        row = self.read_text(text)[0]
        self.assertEqual(row['scenario'], '')
        self.assertEqual(row['ground_truth'], ['T1000'])

    def test_mixed_bad_rows_preserve_ids(self):
        rows = [record(str(i)) for i in range(3)]
        text = '\n'.join(json.dumps(r) if i != 1 else json.dumps(r)[:-1] for i,r in enumerate(rows))
        loaded = self.read_text(text, 3)
        self.assertEqual([r['prompt'] for r in loaded], [r['prompt'] for r in rows])

    def test_incomplete_truth_is_preserved_not_guessed(self):
        text = '{"prompt": "'+PROMPT_START+'\\n\\nlog", "ground_truth": ["T1000"'
        self.assertEqual(self.read_text(text)[0]['ground_truth'], '["T1000"')

    def test_count_mismatch_and_missing_anchor_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.read_text(json.dumps(record()), 990)
        with self.assertRaises(ValueError):
            self.read_text('{"prompt":"unknown", "ground_truth":[]}')

    def test_missing_outer_braces_recovers_prefix_and_unquoted_truth(self):
        text = PROMPT_START+'\\n\\nlog, ground truth: ["T1000"]'
        row = self.read_text(text)[0]
        self.assertEqual(row['ground_truth'], ['T1000'])
        self.assertEqual(row['prompt'], PROMPT_START+'\n\nlog')

    def test_every_returned_string_is_collected_once_unchanged(self):
        outputs = ['', '   ', '{"unterminated":', 'ATT '*3000, '??\n"quoted"\r\ntext']
        for index, raw in enumerate(outputs, 1):
            class Backend:
                context = 4096
                count = staticmethod(lambda text: len(text)//4)
                calls = 0
                def generate(self, prompt, cap):
                    self.calls += 1
                    self.last_prompt = prompt
                    return dict(raw=raw, input_tokens=100, output_tokens=cap, finish='length')
            backend = Backend()
            prompt = record('log with key="unchanged"')['prompt']
            response, metadata = evaluate(backend, prompt, index, self.directory/'attempts.jsonl')
            self.assertEqual(response, raw)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(backend.last_prompt, prompt)
            results = Results(self.directory/f'{index}.csv', [record()], 'config')
            results.begin(1)
            results.update(1, response, **metadata)
            resumed = Results(results.path, [record()], 'config')
            self.assertEqual(resumed.rows[1]['raw response'], raw)
            self.assertEqual(resumed.rows[1]['status'], 'completed')
            self.assertEqual(resumed.pending(), [])

    def test_context_preserves_instructions_and_records_shortening(self):
        prompt = record('abcdefghij'*1000)['prompt']
        effective, plan = fit_prompt(prompt, len, 4096)
        self.assertTrue(plan['input_truncated'])
        self.assertTrue(prompt.startswith(effective))
        self.assertTrue(effective.startswith(PROMPT_START+'\n\n'))
        self.assertLessEqual(len(effective)+plan['max_new_tokens']+16,4096)
        self.assertEqual(plan['original_prompt_tokens'],len(prompt))
        small = record('short')['prompt']
        effective, plan = fit_prompt(small,len,4096)
        self.assertEqual(effective,small)
        self.assertEqual(plan['max_new_tokens'],2048)
        self.assertFalse(plan['input_truncated'])
        long = record('x'*3000)['prompt']
        effective, plan = fit_prompt(long,len,4096)
        self.assertEqual(effective,long)
        self.assertLess(plan['max_new_tokens'],2048)

    def test_unfit_instruction_is_execution_error(self):
        with self.assertRaisesRegex(ValueError,'instruction header'):
            fit_prompt('x'*5000+'\n\nlog',len,4096)

    def test_sparse_resume_and_atomic_failure(self):
        rows = [record(str(i)) for i in range(4)]
        path = self.directory/'rows.csv'
        results = Results(path,rows,'config')
        results.update(3,'')
        results.update(1,'not JSON')
        results.begin(2)
        results.update(2,error='engine unavailable')
        resumed = Results(path,rows,'config')
        self.assertEqual(resumed.pending(),[4,2])
        before = path.read_bytes()
        with patch('camlds.core.os.replace',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                resumed.update(4,'text')
        self.assertEqual(before,path.read_bytes())
        with self.assertRaises(ValueError): Results(path,rows,'changed')
        with self.assertRaises(ValueError): Results(path,[record('different')]*4,'config')

    def test_receiver_checks_identity_not_response(self):
        from camlds.local_store import check_row
        results=Results(self.directory/'rows.csv',[record()],'config')
        for raw in ['', '{bad', 'ATT ATT ATT']:
            results.update(1,raw)
            check_row(results,results.rows[1],'config')
        with self.assertRaises(ValueError):
            check_row(results,{**results.rows[1],'prompt':'changed'},'config')

    def test_same_model_lock_excludes_second_worker(self):
        from filelock import FileLock, Timeout
        with FileLock(self.directory/'model.lock'):
            with self.assertRaises(Timeout):
                with FileLock(self.directory/'model.lock',timeout=0): pass
