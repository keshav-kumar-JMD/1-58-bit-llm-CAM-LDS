"""Run on your computer: python -m camlds.local_store --project .

Listen on loopback; expose only via an HTTPS tunnel you control. The token is
generated locally and never embedded in the notebook. This is a checkpoint
receiver, not a remote shell. Only the known model result paths are writable.
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from .core import Results, atomic_write, digest, load_queries, parse_response, SCHEMA
from .models import MODELS


def serve(project, port=8765):
    project = Path(project).resolve()
    output = project / 'results'
    output.mkdir(exist_ok=True)
    records = load_queries(project / 'data' / 'llm_queries.json')
    expected_hash = digest([{k: v for k, v in r.items() if k != '_recovery'} for r in records])
    token = secrets.token_urlsafe(32)
    print(f'Checkpoint receiver: http://127.0.0.1:{port}\nToken (paste in secret prompt only): {token}', flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log authorization headers or log excerpts.

        def respond(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def dispatch(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                return self.respond(401, {'error': 'Unauthorized'})
            parts = self.path.strip('/').split('/')
            if len(parts) != 2 or parts[1] not in MODELS:
                return self.respond(404, {'error': 'Unknown route'})
            action, model = parts
            manifest_path = output / f'{model}.manifest.json'
            csv_path = output / f'llm_responses_{model}_camlds.csv'
            try:
                if action == 'snapshot' and self.command == 'GET':
                    return self.respond(200, {p.name: p.read_text(encoding='utf-8') for p in (manifest_path, csv_path) if p.exists()})
                size = int(self.headers.get('Content-Length', 0))
                if self.command != 'POST' or not 0 < size < 32_000_000:
                    return self.respond(400, {'error': 'Invalid request size/method'})
                payload = json.loads(self.rfile.read(size))
                if action == 'initialize':
                    manifest = payload['manifest']
                    if manifest['dataset_hash'] != expected_hash or manifest['model'] != model:
                        raise ValueError('Dataset/model mismatch')
                    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                        raise ValueError('Existing manifest differs')
                    results = Results(csv_path, records, manifest['config_hash'])
                    for row in payload['rows']:
                        index = int(row['prompt id'])
                        if index not in results.rows:
                            raise ValueError('Invalid prompt id')
                        if row['status'] == 'ok':
                            check_row(results, row, manifest['config_hash'])
                            results.rows[index] = row
                    results.save()
                    atomic_write(manifest_path, json.dumps(manifest, indent=2))
                elif action == 'row':
                    manifest = json.loads(manifest_path.read_text())
                    results = Results(csv_path, records, manifest['config_hash'])
                    check_row(results, payload, manifest['config_hash'])
                    index = int(payload['prompt id'])
                    if results.rows[index]['status'] != 'ok' or payload['status'] == 'ok':
                        results.rows[index] = payload
                    results.save()
                else:
                    return self.respond(404, {'error': 'Unknown route'})
                return self.respond(200, {'saved': True})
            except (ValueError, KeyError, OSError, TypeError) as error:
                return self.respond(409, {'error': str(error)})

        do_GET = dispatch
        do_POST = dispatch

    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


def check_row(results, row, config_hash):
    index = int(row['prompt id'])
    from .core import BASE_COLUMNS, COLUMNS
    if set(row) != set(COLUMNS) or index not in results.rows or row['config hash'] not in (config_hash, 'legacy-accepted'):
        raise ValueError('Invalid row/schema/config')
    if any(row[key] != results.rows[index][key] for key in BASE_COLUMNS[:6]):
        raise ValueError('Input fields changed')
    if row['status'] == 'ok':
        parse_response(json.dumps({key: json.loads(row[key]) if key == 'techniques' else row[key] for key in SCHEMA['required']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='.')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    serve(args.project, args.port)
