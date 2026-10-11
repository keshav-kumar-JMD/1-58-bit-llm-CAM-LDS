"""Optional authenticated mirror to the physical Windows project directory."""
import json
import os
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse
from .core import atomic_write


class CheckpointTransferError(OSError):
    """Receiver/tunnel failure, distinct from a Colab filesystem failure."""


class Mirror:
    def __init__(self, url, token):
        parsed = urlparse(url)
        if parsed.scheme != 'https' and parsed.hostname not in ('localhost', '127.0.0.1'):
            raise ValueError('Remote checkpoint URL must use HTTPS.')
        self.url, self.token = url.rstrip('/'), token

    @classmethod
    def from_env(cls):
        url, token = os.getenv('CAM_LDS_STORE_URL'), os.getenv('CAM_LDS_STORE_TOKEN')
        if bool(url) != bool(token):
            raise ValueError('Set both checkpoint store URL and token.')
        return cls(url, token) if url else None

    def call(self, route, payload=None):
        request = Request(self.url + route,
            data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
        delays = (2, 5, 10, 20)
        for attempt in range(len(delays) + 1):
            try:
                with urlopen(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as error:
                code = error.code
                try:
                    body = json.loads(error.read(4096).decode('utf-8', errors='replace'))
                    if not isinstance(body, dict):
                        body = {}
                except (ValueError, OSError):
                    body = {}
                finally:
                    error.close()
                reason = f'HTTP {code}: {body.get("error", error.reason)}'
                # Older receivers report temporary file locks as 409. Retrying
                # the same row is idempotent. New receivers explicitly mark
                # genuine data/configuration conflicts as non-retryable.
                retryable = body.get('retryable', code in (408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524))
                hint = ('Rerun Section 3 with the current receiver token.' if code in (401, 403)
                        else 'Check the local receiver/tunnel; close any app locking the results CSV.')
            except (OSError, HTTPException, ValueError) as error:
                reason = f'{type(error).__name__}: {error}'
                retryable = True
                hint = 'Check that the local receiver and HTTPS tunnel are still running.'
            # Avoid leaking a credential if a downstream error includes a URL.
            reason = reason.replace(self.token, '[redacted]') if self.token else reason
            if not retryable or attempt == len(delays):
                raise CheckpointTransferError(
                    f'Checkpoint transfer {route} failed after {attempt + 1} attempt(s): '
                    f'{reason[:800]}. {hint} Existing Colab checkpoints are preserved; '
                    'rerun Section 6 to resume.') from None
            delay = delays[attempt]
            print(f'Checkpoint transfer {route}: {reason[:300]}; retrying in {delay}s '
                  '(the saved response will not be regenerated).', flush=True)
            time.sleep(delay)

    def restore(self, model, output):
        snapshot = self.call('/snapshot/' + model)
        for name, text in snapshot.items():
            # Server only returns these two exact, already validated names.
            allowed = {f'{model}.manifest.json', f'llm_responses_{model}_camlds.csv'}
            if name not in allowed:
                raise ValueError('Unexpected checkpoint filename.')
            target = Path(output) / name
            if target.exists() and target.read_text(encoding='utf-8') != text:
                # Never replace local successes with an older mirror after a
                # network outage. Reconcile during initialize on the server.
                continue
            atomic_write(target, text)

    def initialize(self, model, manifest, results):
        merged = self.call('/initialize/' + model, dict(manifest=manifest, rows=list(results.rows.values())))
        # Restore remote completed rows without replacing newer local successes.
        for row in merged['rows']:
            index = int(row['prompt id'])
            if row['status'] == 'completed' and results.rows[index]['status'] != 'completed':
                results.rows[index] = row
        results.save()

    def row(self, model, row):
        self.call('/row/' + model, row)
