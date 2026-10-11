"""Optional authenticated mirror to the physical Windows project directory."""
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse
from .core import atomic_write


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
        with urlopen(request, timeout=60) as response:
            return json.load(response)

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
        self.call('/initialize/' + model, dict(manifest=manifest, rows=list(results.rows.values())))

    def row(self, model, row):
        self.call('/row/' + model, row)
