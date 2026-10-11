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

from .core import Results, atomic_write, digest, load_queries, fit_prompt, OUTPUT_SUBDIR
from .models import DEFAULT_MODELS, MODELS, gpu_info, load_backend
from .sync import CheckpointTransferError

POLICY = dict(version=2, mode='raw-baseline', prompt='original-no-cleanup-no-added-instructions',
              max_new_tokens=2048, min_new_tokens=256, temperature=0.0,
              repetition_penalty=1.0, seed=42, decoding='greedy-unconstrained',
              context='reduce-output-budget-then-truncate-log-tail', safety_margin=16)


def event(path, **data):
    data['time'] = datetime.now(timezone.utc).isoformat()
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(backend, prompt, record_id, diagnostics):
    effective, plan = fit_prompt(prompt, backend.count, backend.context,
                                 POLICY['max_new_tokens'], POLICY['min_new_tokens'])
    result = backend.generate(effective, plan['max_new_tokens'])
    # No response parsing, screening, repair, or quality-dependent retries.
    if not isinstance(result['raw'], str):
        raise TypeError('Backend did not return response text.')
    event(diagnostics, prompt_id=record_id, status='completed', plan=plan, **result)
    return result['raw'], dict(**{'input tokens': result['input_tokens'],
        'output tokens': result['output_tokens'], 'finish reason': result['finish'],
        'original prompt tokens': plan['original_prompt_tokens'], 'effective prompt': effective,
        'input truncated': plan['input_truncated'], 'max new tokens': plan['max_new_tokens']})


def worker(project, model_key, ids=None, low_memory=False):
    from filelock import FileLock
    from huggingface_hub import HfApi
    from .sync import Mirror
    project = Path(project).resolve()
    output = project / OUTPUT_SUBDIR
    output.mkdir(parents=True, exist_ok=True)
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
            print(f'{model_key}: all requested responses are already collected.', flush=True)
            return
        backend = None
        try:
            backend = load_backend(spec, revision, str(project / '.cache' / 'huggingface'), low_memory=low_memory)
            import importlib.metadata
            versions = {}
            for name in ['llama-cpp-python', 'transformers', 'torch']:
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    pass
            event(diagnostics, status='model_loaded', gpu=gpu_info(), profile=backend.profile,
                  versions=versions, revision=revision)
            atomic_write(output / f'{model_key}.loaded.json', json.dumps({'pid': os.getpid(), 'profile': backend.profile}))
            print(f'{model_key}: {len(pending)} responses remaining.', flush=True)
            for index in pending:
                results.begin(index)  # Crash marker persisted before inference.
                try:
                    response, metadata = evaluate(backend, records[index-1]['prompt'], index, diagnostics)
                except OSError:
                    raise  # A diagnostic/checkpoint write failure must remain visible.
                except Exception as error:
                    message = f'{type(error).__name__}: {error}'
                    results.update(index, error=message)
                    if mirror:
                        mirror.row(model_key, results.rows[index])
                    event(diagnostics, status='execution_error', prompt_id=index, error=message,
                          traceback=traceback.format_exc())
                    print(f'{model_key}: ID {index} execution error: {message}', flush=True)
                    # Reload after a poisoned CUDA context or memory error. Re-load
                    # failures leave untouched rows pending for a fresh worker pass.
                    if isinstance(error, (RuntimeError, MemoryError)):
                        backend.close()
                        backend = None
                        backend = load_backend(spec, revision, str(project / '.cache' / 'huggingface'), low_memory=True)
                    continue
                results.update(index, response, **metadata)
                if mirror:
                    mirror.row(model_key, results.rows[index])
                if index % 10 == 0 or ids is not None:
                    print(f'{model_key}: ID {index} collected; {len(records)-len(results.pending())}/{len(records)}', flush=True)
        finally:
            if backend is not None:
                backend.close()
        print(f'{model_key}: collected={len(records)-len(results.pending())}/{len(records)}; retryable={len(results.pending())}', flush=True)


def summary(project):
    result = []
    for path in sorted((Path(project) / OUTPUT_SUBDIR).glob('llm_responses_*_camlds.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        result.append(dict(file=path.name, rows=len(rows), completed=sum(r['status'] == 'completed' for r in rows),
                           errors={int(r['prompt id']): r['error'] for r in rows if r['status'] != 'completed' and r.get('error')},
                           retryable_ids=[int(r['prompt id']) for r in rows if r['status'] != 'completed']))
    return result


def print_report(report, ids=None):
    """Show response coverage without dumping 990 pending IDs."""
    requested = set(ids or range(1, 991))
    for row in report['coverage']:
        remaining = sorted(requested.intersection(row['retryable_ids']))
        print(f'{row["file"]}: {len(requested)-len(remaining)}/{len(requested)} responses collected; '
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
    output = project / OUTPUT_SUBDIR
    output.mkdir(parents=True, exist_ok=True)
    active = {}
    queue = [(key, 0, set()) for key in selected]
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
                key, retry, skip = queue[0]
                free = gpu_info()['free_gib']
                # Cross-notebook supervisors serialize admission, then wait until
                # the next polling round so free memory reflects loaded workers.
                if not active or free > MODELS[key]['vram_gib'] + 4:
                    queue.pop(0)
                    log = (output / f'{key}.worker.log').open('a', encoding='utf-8')
                    command = [python, '-u', '-m', 'camlds.runner', '--project', str(project), '--model', key]
                    requested = set(ids or range(1, 991)) - skip
                    command += ['--ids', ','.join(map(str, sorted(requested)))]
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
                    active[key] = (process, log, retry, skip)
                    print(f'Started {key} (PID {process.pid}); log: {log.name}', flush=True)
            for key, (process, log, retry, skip) in list(active.items()):
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
                if code == 74:
                    raise RuntimeError(f'{key}: local receiver transfer failed after retries. '
                        f'Colab checkpoints are preserved. Receiver details:\n{log_tail(log.name, 1800)}')
                coverage = next((r for r in summary(project) if r['file'] == f'llm_responses_{key}_camlds.csv'), None)
                missing = coverage is None or bool(set(coverage['retryable_ids']) & set(ids or range(1, 991)))
                # Native crashes leave the current row marked running. Quarantine
                # it for this pass so one bad row cannot prevent later rows running.
                if code != 0 and coverage:
                    path = output / f'llm_responses_{key}_camlds.csv'
                    records = load_queries(project / 'data/llm_queries.json')
                    manifest = json.loads((output / f'{key}.manifest.json').read_text())
                    checkpoint = Results(path, records, manifest['config_hash'])
                    crashed = {i for i, row in checkpoint.rows.items() if row['status'] == 'running'}
                    for index in crashed:
                        checkpoint.update(index, error=f'Worker exited {code} during generation; no response returned.')
                    if crashed:
                        skip = skip | crashed
                        if set(checkpoint.pending()) & (set(ids or range(1, 991)) - skip):
                            queue.append((key, retry, skip))
                            continue
                if (code != 0 or missing) and retry + 1 < passes:
                    queue.append((key, retry + 1, set()))
                elif code != 0 or missing:
                    failures[key] = f'Execution failures remain after {passes} passes; inspect {key}.worker.log and {key}.attempts.jsonl'
                print(f'{key} exited {code}; collected={coverage["completed"] if coverage else 0}/990', flush=True)
            if active and time.monotonic() - last_progress >= 30:
                for row in summary(project):
                    if any(row['file'] == f'llm_responses_{key}_camlds.csv' for key in active):
                        print(f'{row["file"]}: {row["completed"]}/990 collected; {len(row["retryable_ids"])} remaining', flush=True)
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
    except CheckpointTransferError:
        traceback.print_exc()
        sys.exit(74)
    except OSError:
        traceback.print_exc()
        sys.exit(73)  # Supervisor stops all work; never disguise a save failure.
