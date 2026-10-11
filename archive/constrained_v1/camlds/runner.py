"""Isolated workers, resumable rows, diagnostics and same-runtime scheduling."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .core import Results, aggregate, atomic_write, clean_prompt, digest, load_queries, parse_response, plan_chunks, SCHEMA
from .models import DEFAULT_MODELS, MODELS, gpu_info, load_backend

POLICY = dict(version=1, schema=SCHEMA, output_caps=[768, 1024, 1024],
              temperatures=[0.0, 0.2, 0.2], repetition=[1.12, 1.2, 1.2], seed=42,
              chunking='all-log-text; reciprocal-rank-60; max-attack-confidence')


def event(path, **data):
    data['time'] = datetime.now(timezone.utc).isoformat()
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(backend, prompt, record_id, diagnostics, previous_attempts=0):
    chunks = plan_chunks(clean_prompt(prompt), backend.count, backend.context, max(POLICY['output_caps']))
    responses, input_tokens, output_tokens, attempts = [], 0, 0, previous_attempts
    for chunk_index, chunk in enumerate(chunks, 1):
        last_error = ''
        for retry, cap in enumerate(POLICY['output_caps']):
            attempts += 1
            result = {}
            try:
                result = backend.generate(chunk, cap, retry)
                input_tokens += result['input_tokens']
                output_tokens += result['output_tokens']
                # A complete valid JSON at the cap is acceptable. Do not reject
                # solely because the engine reported finish_reason=length.
                response = parse_response(result['raw'])
                event(diagnostics, prompt_id=record_id, chunk=chunk_index, attempt=attempts,
                      status='ok', **result)
                responses.append(response)
                break
            except (ValueError, TypeError, KeyError) as error:
                last_error = f'{type(error).__name__}: {error}'
                event(diagnostics, prompt_id=record_id, chunk=chunk_index, attempt=attempts,
                      status='invalid_output', error=last_error, **result)
        else:
            return None, dict(error=last_error, attempts=attempts, chunks=len(chunks),
                              **{'input tokens': input_tokens, 'output tokens': output_tokens,
                                 'finish reason': result.get('finish', 'error')})
    return aggregate(responses), dict(attempts=attempts, chunks=len(chunks),
            **{'input tokens': input_tokens, 'output tokens': output_tokens, 'finish reason': 'validated'})


def worker(project, model_key, ids=None, low_memory=False):
    from filelock import FileLock
    from huggingface_hub import HfApi
    from .sync import Mirror
    project = Path(project).resolve()
    output = project / 'results'
    output.mkdir(exist_ok=True)
    records = load_queries(project / 'data' / 'llm_queries.json')
    spec = MODELS[model_key]
    diagnostics = output / f'{model_key}.attempts.jsonl'
    mirror = Mirror.from_env()
    # This lock is released by the OS after a crash; do not use PID sentinel files.
    with FileLock(str(output / f'{model_key}.lock'), timeout=0):
        manifest_path = output / f'{model_key}.manifest.json'
        csv_path = output / f'llm_responses_{model_key}_camlds.csv'
        if mirror:
            mirror.restore(model_key, output)
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        dataset_hash = digest([{k: v for k, v in r.items() if k != '_recovery'} for r in records])
        if manifest:
            if manifest['dataset_hash'] != dataset_hash or manifest['spec'] != spec or manifest['policy'] != POLICY:
                raise ValueError('Dataset/model/policy differs from saved manifest. Preserve this run and choose another project for a new experiment.')
            revision = manifest['revision']
        else:
            revision = HfApi().model_info(spec['repo']).sha
            manifest = dict(model=model_key, spec=spec, revision=revision,
                            dataset_hash=dataset_hash, policy=POLICY)
            manifest['config_hash'] = digest(manifest)
            atomic_write(manifest_path, json.dumps(manifest, indent=2))
        results = Results(csv_path, records, manifest['config_hash'])
        if mirror:
            mirror.initialize(model_key, manifest, results)
        pending = [i for i in results.pending() if ids is None or i in ids]
        if not pending:
            print(f'{model_key}: all requested rows are already valid.', flush=True)
            return
        backend = None
        try:
            backend = load_backend(spec, revision, str(project / '.cache' / 'huggingface'), low_memory=low_memory)
            import importlib.metadata
            versions = {}
            for name in ['llama-cpp-python', 'transformers', 'torch', 'lm-format-enforcer']:
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    pass
            event(diagnostics, status='model_loaded', gpu=gpu_info(), profile=backend.profile,
                  versions=versions, revision=revision)
            atomic_write(output / f'{model_key}.loaded.json', json.dumps({'pid': os.getpid(), 'profile': backend.profile}))
            audit = []
            for index, record in enumerate(records, 1):
                chunks = plan_chunks(clean_prompt(record['prompt']), backend.count, backend.context, 1024)
                audit.append(dict(prompt_id=index, full_tokens=backend.count(clean_prompt(record['prompt'])),
                                  chunk_tokens=[backend.count(c) for c in chunks], chunks=len(chunks)))
            atomic_write(output / f'{model_key}.token_audit.json', json.dumps(audit, indent=2))
            print(f'{model_key}: {len(pending)} missing/failed; {sum(a["chunks"] > 1 for a in audit)} rows need chunks.', flush=True)
            for index in pending:
                old_attempts = int(results.rows[index]['attempts'])
                try:
                    response, metadata = evaluate(backend, records[index-1]['prompt'], index, diagnostics, old_attempts)
                except (RuntimeError, MemoryError) as error:
                    # A CUDA assertion may poison this process. Persist then let
                    # the supervisor start a fresh worker, not a new notebook VM.
                    results.update(index, error=f'{type(error).__name__}: {error}', attempts=old_attempts+1)
                    if mirror:
                        mirror.row(model_key, results.rows[index])
                    event(diagnostics, status='engine_error', prompt_id=index, error=str(error), traceback=traceback.format_exc())
                    raise
                # Keep persistence OUTSIDE inference exception handling: disk or
                # mirror failure must stop the worker instead of being swallowed.
                results.update(index, response, **metadata)
                if mirror:
                    mirror.row(model_key, results.rows[index])
                if index % 10 == 0 or response is None:
                    print(f'{model_key}: id={index}, status={results.rows[index]["status"]}, valid={len(records)-len(results.pending())}/{len(records)}', flush=True)
                    if response is None:
                        print(f'  Validation error: {results.rows[index]["error"]}', flush=True)
        finally:
            if backend is not None:
                backend.close()
        print(f'{model_key}: valid={len(records)-len(results.pending())}/{len(records)}; retryable={len(results.pending())}', flush=True)


def summary(project):
    result = []
    for path in sorted((Path(project) / 'results').glob('llm_responses_*_camlds.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        result.append(dict(file=path.name, rows=len(rows), valid=sum(r['status'] == 'ok' for r in rows),
                           errors={int(r['prompt id']): r['error'] for r in rows if r['status'] != 'ok' and r.get('error')},
                           retryable_ids=[int(r['prompt id']) for r in rows if r['status'] != 'ok']))
    return result


def print_report(report, ids=None):
    """Show requested smoke-test coverage without dumping 990 pending IDs."""
    requested = set(ids or range(1, 991))
    for row in report['coverage']:
        remaining = sorted(requested.intersection(row['retryable_ids']))
        print(f'{row["file"]}: {len(requested)-len(remaining)}/{len(requested)} requested rows valid; '
              f'{len(remaining)} remaining' + (f' {remaining[:10]}' if remaining else ''))
        for index in remaining[:10]:
            if row.get('errors', {}).get(index):
                print(f'  ID {index}: {row["errors"][index]}')
    for model, message in report['incomplete'].items():
        print(f'{model}: {message}')


def log_tail(path, limit=4000):
    with Path(path).open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read().decode('utf-8', errors='replace')


def run_suite(project, python=None, selected=None, max_parallel=1, ids=None, passes=2):
    """Schedule independent processes on one GPU. A second notebook may select
    different model keys on the SAME filesystem/runtime. Model locks prevent
    duplicate work; estimates are admission heuristics, not memory guarantees.
    """
    from filelock import FileLock
    project = Path(project).resolve()
    python = str(python or sys.executable)
    selected = list(DEFAULT_MODELS if selected is None else selected)
    if len(set(selected)) != len(selected) or not selected or set(selected) - MODELS.keys():
        raise ValueError('Choose distinct known model keys.')
    if max_parallel < 1 or passes < 1:
        raise ValueError('max_parallel and passes must be positive.')
    output = project / 'results'
    output.mkdir(exist_ok=True)
    active = {}
    queue = [(key, 0) for key in selected]
    failures = {}
    last_progress = time.monotonic()
    try:
        while queue or active:
            # Avoid admitting a second large model while the first is still
            # downloading/loading and has not allocated its VRAM yet.
            loading = any(not (output / f'{key}.loaded.json').exists() or
                          json.loads((output / f'{key}.loaded.json').read_text()).get('pid') != values[0].pid
                          for key, values in active.items() if values[0].poll() is None)
            if queue and len(active) < max_parallel and not loading:
                key, retry = queue[0]
                free = gpu_info()['free_gib']
                # Cross-notebook supervisors serialize admission, then wait until
                # the next polling round so free memory reflects loaded workers.
                if not active or free > MODELS[key]['vram_gib'] + 4:
                    queue.pop(0)
                    log = (output / f'{key}.worker.log').open('a', encoding='utf-8')
                    command = [python, '-u', '-m', 'camlds.runner', '--project', str(project), '--model', key]
                    if ids:
                        command += ['--ids', ','.join(map(str, ids))]
                    if retry:
                        command += ['--low-memory']
                    env = os.environ.copy()
                    # Disable architecture-specific Torch compilation by default;
                    # official packed BitNet weights are unpacked by HF on load.
                    env['TORCHDYNAMO_DISABLE'] = '1'
                    env['TOKENIZERS_PARALLELISM'] = 'false'
                    env['PYTHONFAULTHANDLER'] = '1'
                    if MODELS[key].get('backend') != 'transformers':
                        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
                    env['PYTHONPATH'] = str(project) + os.pathsep + env.get('PYTHONPATH', '')
                    with FileLock(str(output / 'admission.lock')):
                        process = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[key] = (process, log, retry, time.monotonic())
                    print(f'Started {key} (PID {process.pid}); log: {log.name}', flush=True)
            for key, (process, log, retry, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[key]
                if code != 0:
                    try:
                        reason = signal.Signals(-code).name if code < 0 else f'exit {code}'
                    except ValueError:
                        reason = f'exit {code}'
                    print(f'{key}: worker failed ({reason}). Last log output:\n{log_tail(log.name)}', flush=True)
                if code == 73:
                    raise RuntimeError(f'{key}: storage/checkpoint I/O failed; inspect {key}.worker.log. Resolve it before resuming.')
                coverage = next((r for r in summary(project) if r['file'] == f'llm_responses_{key}_camlds.csv'), None)
                missing = coverage is None or bool(set(coverage['retryable_ids']) & set(ids or range(1, 991)))
                if (code != 0 or missing) and retry + 1 < passes:
                    queue.append((key, retry + 1))
                elif code != 0 or missing:
                    failures[key] = f'Incomplete after {passes} passes; inspect {key}.worker.log and attempts.jsonl'
                print(f'{key} exited {code}; valid={coverage["valid"] if coverage else 0}/990', flush=True)
            if active and time.monotonic() - last_progress >= 30:
                for row in summary(project):
                    if any(row['file'] == f'llm_responses_{key}_camlds.csv' for key in active):
                        print(f'{row["file"]}: {row["valid"]}/990 valid; {len(row["retryable_ids"])} remaining', flush=True)
                last_progress = time.monotonic()
            if active:
                # Give new loads time to allocate before admitting another worker.
                time.sleep(5)
    finally:
        for process, log, _, _ in active.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    return dict(coverage=summary(project), incomplete=failures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--ids')
    parser.add_argument('--low-memory', action='store_true')
    args = parser.parse_args()
    ids = set(map(int, args.ids.split(','))) if args.ids else None
    if ids is not None and (not ids or min(ids) < 1 or max(ids) > 990):
        parser.error('IDs must be in 1..990')
    try:
        worker(args.project, args.model, ids, args.low_memory)
    except OSError:
        traceback.print_exc()
        sys.exit(73)  # Supervisor stops all work; never disguise a save failure.
