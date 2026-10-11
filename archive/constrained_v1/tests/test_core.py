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

    def test_chunking_covers_every_log_character_and_respects_budget(self):
        log = ('x'*701+'\nUnicode café 😀\n') * 8
        original = record(log)['prompt']
        count = lambda s: len(s)+11  # simulated chat framing
        chunks = plan_chunks(original, count, 2048, 1024)
        prefix = PROMPT_START + INSTRUCTION + '\n\n'
        self.assertEqual(''.join(c[len(prefix):] for c in chunks), log)
        self.assertTrue(all(count(c)+1024+32 <= 2048 for c in chunks))

    def test_response_rejects_saved_failure_patterns(self):
        for bad in ['{"techniques":["ATT", "ATT"]', json.dumps({**valid(), 'techniques':['ATT']*10}),
                    json.dumps({**valid(), 'techniques':['T1000']*10}),
                    json.dumps({**valid(), 'explanation':'loop '*100})]:
            with self.assertRaises(ValueError):
                parse_response(bad)
        self.assertEqual(parse_response(json.dumps(valid())), valid())

    def test_sparse_resume_fills_earlier_gaps_without_duplicates(self):
        path = self.directory / 'results.csv'
        records = [record(str(i)) for i in range(5)]
        results = Results(path, records, 'config')
        results.update(1, valid(), attempts=1)
        results.update(4, valid(), attempts=1)
        results.update(2, error='bad JSON', attempts=3)
        resumed = Results(path, records, 'config')
        self.assertEqual(resumed.pending(), [2,3,5])
        resumed.update(2, valid(), attempts=4)
        self.assertEqual(Results(path, records, 'config').pending(), [3,5])
        with path.open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 5)

    def test_atomic_interruption_keeps_original(self):
        path = self.directory / 'results.csv'
        results = Results(path, [record('quote "\nnewline café')], 'config')
        before = path.read_bytes()
        with patch('camlds.core.os.replace', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                results.update(1, valid())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(Results(path, [record('quote "\nnewline café')], 'config').pending(), [1])

    def test_changed_input_and_config_are_rejected(self):
        path = self.directory / 'results.csv'
        results = Results(path, [record()], 'old')
        results.update(1, valid())
        with self.assertRaises(ValueError):
            Results(path, [record('different')], 'old')
        with self.assertRaises(ValueError):
            Results(path, [record()], 'new')

    def test_retry_then_accept_valid_output_even_at_token_cap(self):
        class Backend:
            context = 4096
            count = staticmethod(lambda text: len(text)//4)
            def generate(self, prompt, cap, attempt):
                return dict(raw='{"unfinished":' if attempt == 0 else json.dumps(valid()),
                            input_tokens=100, output_tokens=cap, finish='length')
        response, metadata = evaluate(Backend(), record()['prompt'], 1, self.directory/'diagnostics.jsonl')
        self.assertEqual(response, valid())
        self.assertEqual(metadata['attempts'], 2)
        self.assertEqual(len((self.directory/'diagnostics.jsonl').read_text().splitlines()), 2)

    def test_exhausted_retries_return_explicit_failure(self):
        class Backend:
            context = 4096
            count = staticmethod(lambda text: len(text)//4)
            def generate(self, *args):
                return dict(raw='bad', input_tokens=10, output_tokens=20, finish='length')
        response, metadata = evaluate(Backend(), record()['prompt'], 2, self.directory/'fail.jsonl')
        self.assertIsNone(response)
        self.assertEqual(metadata['attempts'], 3)
        self.assertTrue(metadata['error'])

    def test_ground_truth_never_sent(self):
        class Backend:
            context = 4096
            count = staticmethod(lambda text: len(text)//4)
            def generate(self, prompt, *args):
                assert 'secret-ground-truth' not in prompt
                return dict(raw=json.dumps(valid()), input_tokens=10, output_tokens=20, finish='stop')
        r = record()
        r['ground_truth'] = ['secret-ground-truth']
        evaluate(Backend(), r['prompt'], 1, self.directory/'truth.jsonl')

    def test_same_model_lock_excludes_second_worker(self):
        from filelock import FileLock, Timeout
        path = self.directory/'model.lock'
        with FileLock(path):
            with self.assertRaises(Timeout):
                with FileLock(path, timeout=0):
                    pass

    def test_local_mirror_row_validation(self):
        from camlds.local_store import check_row
        results = Results(self.directory/'rows.csv', [record()], 'config')
        results.update(1, valid())
        check_row(results, results.rows[1], 'config')
        bad = {**results.rows[1], 'prompt':'changed'}
        with self.assertRaises(ValueError):
            check_row(results, bad, 'config')

    def test_migrate_legacy_keeps_good_predictions_retries_invalid(self):
        records = [record('one'), record('two')]
        source, destination = self.directory/'legacy.csv', self.directory/'new.csv'
        with source.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(BASE_COLUMNS)
            for index, r in enumerate(records, 1):
                response = valid() if index == 1 else {**valid(), 'techniques':['ATT']*10}
                writer.writerow([index]+[csv_cell(r[k]) for k in INPUT_KEYS]+[csv_cell(response[k]) for k in SCHEMA['required']])
        import_legacy(source, destination, records)
        resumed = Results(destination, records, 'new-policy')
        self.assertEqual(resumed.pending(), [2])
        self.assertEqual(resumed.rows[1]['config hash'], 'legacy-accepted')


if __name__ == '__main__':
    unittest.main()
