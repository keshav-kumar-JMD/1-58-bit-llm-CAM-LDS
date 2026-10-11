"""Input recovery, context fitting, and durable raw-response CSVs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

PROMPT_START = 'You are a MITRE ATT&CK TTP classification expert.'
INPUT_KEYS = ['scenario', 'variant', 'step', 'prompt', 'ground_truth']
BASE_COLUMNS = ['prompt id', 'scenario', 'variant', 'step', 'prompt', 'ground truth']
COLUMNS = BASE_COLUMNS + ['raw response', 'status', 'attempts', 'error',
                          'input tokens', 'output tokens', 'finish reason',
                          'original prompt tokens', 'effective prompt', 'input truncated',
                          'max new tokens', 'input recovery', 'config hash']
OUTPUT_SUBDIR = 'results/raw_baseline'
csv.field_size_limit(2**31 - 1)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def atomic_write(path, text):
    """Replacement in the same directory leaves either the old or new full file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _decode_fragment(text):
    """Decode JSON escapes without interpreting unknown escapes or losing Unicode."""
    def replace(match):
        try:
            return json.loads('"' + match[0] + '"')
        except ValueError:
            return match[0]
    return re.sub(r'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt])', replace, text)


def _recover_record(raw):
    # Field anchor is unescaped; log data containing an escaped key is not a boundary.
    start = raw.find(PROMPT_START)
    ground = list(re.finditer(r'(?<![\\\w])(?:["\'])?ground[ _-]*truth(?:["\'])?\s*:', raw, re.I))
    if start < 0 or not ground or ground[-1].start() <= start:
        raise ValueError('Cannot recover record: need the known prompt prefix and ground truth field.')
    end = ground[-1].start()
    prompt = raw[start:end].rstrip()
    prompt = re.sub(r'["\']?\s*,?\s*$', '', prompt)
    truth_text = raw[ground[-1].end():].strip()
    try:
        truth, _ = json.JSONDecoder().raw_decode(truth_text)
    except ValueError:
        truth = _decode_fragment(truth_text.rstrip(' \r\n},'))
    record = {'prompt': _decode_fragment(prompt), 'ground_truth': truth}
    for key in ['scenario', 'variant', 'step']:
        match = re.search(r'["\']' + key + r'["\']\s*:\s*("(?:\\.|[^"\\])*"|\d+)', raw[:start])
        record[key] = json.loads(match[1]) if match else ''
    record['_recovery'] = 'Recovered malformed outer JSON; inspect original ground truth.'
    return record


def load_queries(path, expected=990):
    """Prefer strict array/JSONL parsing, then recover by record/prompt anchors.

    Log JSON inside a prompt is always opaque text. Never parse or repair it.
    Missing optional metadata does not remove a record. Ambiguous input fails visibly.
    """
    text = Path(path).read_text(encoding='utf-8-sig')
    records = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = [parsed]
    except ValueError:
        pass
    if records is None:
        lines = [line for line in text.splitlines() if line.strip()]
        try:
            records = [json.loads(line) for line in lines]
        except ValueError:
            # The dataset has one prompt prefix per record. Splitting at the
            # outer object preserves metadata even if a row has literal newlines.
            anchors = list(re.finditer(r'(?m)^\s*\{\s*["\'](?:scenario|prompt)["\']\s*:', text))
            prompt_starts = list(re.finditer(re.escape(PROMPT_START), text))
            if not anchors or len(anchors) != len(prompt_starts):
                # Last-resort record framing for missing braces/field labels.
                # Optional metadata cannot safely be associated in this mode;
                # recover prompt/truth only, in unchanged source order.
                anchors = prompt_starts
            if not anchors:
                raise ValueError('No known prompt boundaries found.')
            blocks = [text[m.start():anchors[i+1].start() if i+1 < len(anchors) else len(text)]
                      for i, m in enumerate(anchors)]
            records = []
            for block in blocks:
                try:
                    records.append(json.loads(block.strip().rstrip(',')))
                except ValueError:
                    records.append(_recover_record(block))
    normalized = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f'Record {index} is not an object.')
        truth_key = next((key for key in record if re.fullmatch(r'ground[ _-]*truth', key, re.I)), None)
        if not isinstance(record.get('prompt'), str) or not record['prompt'].startswith(PROMPT_START) or truth_key is None:
            raise ValueError(f'Record {index}: missing known prompt prefix or ground truth field.')
        item = {key: record.get(key, '') for key in INPUT_KEYS}
        item['ground_truth'] = record[truth_key]
        item['_recovery'] = record.get('_recovery', '')
        normalized.append(item)
    if expected is not None and len(normalized) != expected:
        raise ValueError(f'Expected {expected} records, found {len(normalized)}; refusing to shift source IDs.')
    return normalized


def fit_prompt(prompt, count_tokens, context, max_new_tokens=2048, min_new_tokens=256):
    """One unchanged prompt where possible; shorten only to meet the hard limit.

    Long inputs first use the available output budget (down to 256 tokens).
    If input still cannot fit, retain the complete original instruction header
    and the largest measured leading log excerpt. No reminders or chunk fusion.
    """
    original_tokens = count_tokens(prompt)
    budget = context - min_new_tokens - 16
    effective = prompt
    if original_tokens > budget:
        header, separator, logs = prompt.partition('\n\n')
        if not separator or count_tokens(header + separator) > budget:
            raise ValueError('Original instruction header does not fit the model context.')
        low, high = 0, len(logs)
        # Token counts are not strictly monotonic; always check the chosen result.
        while low < high:
            mid = (low + high + 1) // 2
            if count_tokens(header + separator + logs[:mid]) <= budget:
                low = mid
            else:
                high = mid - 1
        effective = header + separator + logs[:low]
    input_tokens = count_tokens(effective)
    cap = min(max_new_tokens, context - input_tokens - 16)
    if input_tokens > budget or cap < min_new_tokens:
        raise ValueError('Unable to fit prompt with reserved output space.')
    return effective, dict(original_prompt_tokens=original_tokens,
                           input_tokens=input_tokens, max_new_tokens=cap,
                           input_truncated=effective != prompt)


def csv_cell(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)


class Results:
    """Atomic raw-response checkpoint. Completion says nothing about quality."""
    def __init__(self, path, records, config_hash):
        self.path, self.records, self.config_hash = Path(path), records, config_hash
        self.rows = {}
        for index, record in enumerate(records, 1):
            row = dict.fromkeys(COLUMNS, '')
            row.update(dict(zip(BASE_COLUMNS, [str(index)] + [csv_cell(record[k]) for k in INPUT_KEYS])))
            row.update(status='pending', attempts='0', **{'input recovery': record['_recovery'], 'config hash': config_hash})
            self.rows[index] = row
        if self.path.exists():
            with self.path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream, strict=True)
                if reader.fieldnames != COLUMNS:
                    raise ValueError('Not a raw-baseline CSV. Keep constrained results in their original folder.')
                seen = set()
                for row in reader:
                    index = int(row['prompt id'])
                    if index in seen or index not in self.rows or None in row or any(v is None for v in row.values()):
                        raise ValueError('Duplicate/out-of-range ID or damaged CSV.')
                    seen.add(index)
                    if any(row[k] != self.rows[index][k] for k in BASE_COLUMNS):
                        raise ValueError(f'Input changed for saved prompt {index}.')
                    if row['config hash'] != config_hash:
                        raise ValueError('Experiment settings changed; preserve this baseline before changing settings.')
                    if row['status'] not in ('pending', 'running', 'completed', 'execution_error'):
                        raise ValueError('Unknown checkpoint state.')
                    self.rows[index] = row
        self.save()

    def save(self):
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(self.rows.values())
        atomic_write(self.path, '\ufeff' + buffer.getvalue())

    def pending(self):
        return sorted((i for i, r in self.rows.items() if r['status'] != 'completed'),
                      key=lambda i: (int(self.rows[i]['attempts']), i))

    def begin(self, index):
        self.rows[index].update(status='running', error='', attempts=str(int(self.rows[index]['attempts']) + 1))
        self.save()

    def update(self, index, response=None, **metadata):
        row = self.rows[index]
        row.update({key: str(value) for key, value in metadata.items()})
        row['status'] = 'completed' if response is not None else 'execution_error'
        if response is not None:
            # An empty string, invalid JSON, or a token-limit response is success.
            row['raw response'] = response
            row['error'] = ''
        self.save()
