"""Build a self-contained notebook from readable, testable Python helpers."""
import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(text).strip().splitlines(True)))


def code(text, hidden=False):
    cells.append(dict(cell_type='code', metadata={'jupyter': {'source_hidden': True}} if hidden else {},
                      execution_count=None, outputs=[], source=textwrap.dedent(text).strip().splitlines(True)))


md('''
# CAM-LDS: raw-response baseline on Colab

Run all **990 source queries once per model**, saving whatever text the model
returns. No grammar, JSON enforcement, duplicate filtering, output validation,
answer repair, added prompt instructions, prompt cleanup, or quality-based retries.
Original prompts already contain task instructions; those are preserved verbatim.
Only the model's normal chat template wraps the input. Ground truth stays in the
CSV and is never added to the inference message.

**A completed response is not necessarily correct, meaningful, or well formed.**
Empty text, repetitions, refusals, malformed JSON, and token-limit endings all count
as collected responses. Only execution failures remain eligible for retry.

Stable files live in **`results/raw_baseline/llm_responses_<model>_camlds.csv`**.
The previous constrained outputs remain untouched in `results/`; they are not
baseline observations. The previous notebook/code is preserved under
`archive/constrained_v1/`. The original notebook also remains in `archive/`.

Workflow: settings -> portable helpers -> local receiver -> GPU setup -> input check
-> full 990-row run -> coverage. There is no smoke-test gate.
''')
md('''
## 1. Settings and execution location

Select the best GPU currently offered by Colab in VS Code. Pro+ does not guarantee
H100 availability or a fixed number of concurrent runtimes. Detection uses the
actual GPU name, free VRAM, compute capability and CUDA compiler, including a G4
allocation exposing a Blackwell GPU. A provider-side GPU change may replace the
VM; notebook code cannot prevent that. Restore saved CSVs and rerun setup on the
replacement VM. Ordinary backend rebuilding/reloading uses subprocesses and
does not require restarting the notebook kernel.

**Can be done locally once:** editing code, packaging this notebook, inspecting
the dataset, tests, and the checkpoint receiver. None require a GPU.
''')
code('''
from pathlib import Path
import os, platform, sys

PROJECT_DIR = Path('/content/CAM_LDS') if platform.system() == 'Linux' else Path.cwd()
SELECTED_MODELS = ['qwen1b', 'hermes', 'bitnet2b', 'llama3b', 'mistral7b', 'qwen7b']
MAX_PARALLEL_MODELS = 1  # One model at a time is the simplest default.
RETRY_PASSES = 2        # Retry execution errors only; never regenerate a collected response.
REQUIRE_LOCAL_MIRROR = True  # Recommended for your VS Code -> Colab setup.
# Optional NEW experiment only (disabled to keep resume stable):
# from datetime import datetime, timezone
# PROJECT_DIR = PROJECT_DIR / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

PROJECT_DIR.mkdir(parents=True, exist_ok=True)
print('Kernel host:', platform.node(), '| Kernel project:', PROJECT_DIR)
print('Your Windows project is a separate filesystem when this kernel is on Colab.')
''')
md('''
## 2. Restore portable source and the exact attached dataset

Run this once per kernel/session, before any workers start. The collapsed cell
contains a compressed copy of `camlds/` and `data/llm_queries.json`; it does not
download another dataset revision. It never replaces an existing different data
file. The helper files are regenerated from the readable project sources using
`python tools/build_notebook.py`.

**Can be prepared locally once:** the embedded code/data bundle. **Model downloads
are cached by revision:** repeat runs on the same VM reuse `.cache/huggingface`.
For a new Colab VM, transfer that cache using VS Code's upload/download controls
if you want to avoid downloading again. Windows compiled binaries cannot be reused
on Colab Linux. Compatible Linux wheels can be transferred with `.cache/wheels`.
''')
bundle = io.BytesIO()
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for path in sorted((ROOT / 'camlds').glob('*.py')):
        z.write(path, path.relative_to(ROOT).as_posix())
    z.write(ROOT/'data/llm_queries.json', 'data/llm_queries.json')
payload = base64.b64encode(bundle.getvalue()).decode()
code('''
import base64, io, zipfile, hashlib
PAYLOAD = "''' + payload + '''"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as bundle:
    for item in bundle.infolist():
        target = (PROJECT_DIR / item.filename).resolve()
        if not target.is_relative_to(PROJECT_DIR.resolve()):
            raise ValueError('Unsafe bundle path')
        data = bundle.read(item)
        if item.filename.startswith('data/') and target.exists() and target.read_bytes() != data:
            raise ValueError('Existing query file differs from the bundled dataset; preserve it before starting a new experiment.')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
sys.path.insert(0, str(PROJECT_DIR))
# Refresh notebook imports after a helper update; workers are new processes.
import importlib
for module_name in list(sys.modules):
    if module_name == 'camlds' or module_name.startswith('camlds.'):
        del sys.modules[module_name]
importlib.invalidate_caches()
print('Portable helpers ready; query SHA256:', hashlib.sha256((PROJECT_DIR/'data/llm_queries.json').read_bytes()).hexdigest())
del PAYLOAD
''', hidden=True)
md('''
## 3. Connect the local raw-response checkpoint receiver

VS Code's Colab kernel runs remotely; `Path.cwd()` is not your Windows workspace.
For automatic local writes, start these in **two LOCAL VS Code terminals**, with
this project as the working directory:

```powershell
python -m camlds.local_store --project .
```

```powershell
.\\tools\\cloudflared.exe tunnel --url http://127.0.0.1:8765
```

**Local one-time setup:** install `cloudflared` following its
[official download instructions](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/).
The receiver needs only Python's standard library. An existing HTTPS tunnel you
control works too. Paste the tunnel's HTTPS URL and the receiver's token in the
prompts below. The token is kept in process environment, not notebook source or
saved output. The receiver binds to loopback and accepts authenticated writes only
to known model checkpoint paths. Keep both local terminals running; no Drive mount
is used. The tunnel transports dataset/result content to your local receiver.

**After updating from the constrained notebook, stop and restart the first terminal**
so the receiver loads the raw-baseline protocol. This prints a new token; rerun
this section with that token. The tunnel can stay running. Outputs go to
`results/raw_baseline` on both computers.

Every completed row is first saved on Colab, then acknowledged by the local
receiver. Temporary connection/file-lock failures retry the same saved checkpoint
up to five times, waiting 2, 5, 10 and 20 seconds between attempts. This does not
regenerate the model response. Persistent errors stop with the receiver's reason;
rerunning reconciles saved successes. New sessions restore CSVs and revision pins.
Run one worker per model; different models can use this receiver concurrently.

If you prefer manual transfer, set `REQUIRE_LOCAL_MIRROR=False`, then download
`results/` from the Colab server using VS Code into **this project's `results/`**.
Upload it to `PROJECT_DIR/results/` before the next session. Colab local disk is
ephemeral, so this mode requires your exports to preserve work across VM deletion.
''')
code('''
from getpass import getpass
from camlds.sync import Mirror
from urllib.parse import urlparse

# Always ask again when rerunning this cell. Save credentials only after a
# successful connection, so a failed attempt cannot trap us with a stale URL.
os.environ.pop('CAM_LDS_STORE_URL', None)
os.environ.pop('CAM_LDS_STORE_TOKEN', None)
if REQUIRE_LOCAL_MIRROR:
    receiver_url = input('Local receiver HTTPS tunnel URL: ').strip()
    parsed_url = urlparse(receiver_url)
    if parsed_url.scheme != 'https' or not parsed_url.hostname:
        raise ValueError('Paste the complete https://...trycloudflare.com URL printed by cloudflared, then rerun this cell.')
    receiver_token = getpass('Local receiver token: ').strip()
    if not receiver_token:
        raise ValueError('Receiver token is empty; rerun this cell and paste the token from your first local terminal.')
    try:
        health = Mirror(receiver_url, receiver_token).call('/health/qwen1b')
        if health.get('protocol') != 'raw-baseline-v2':
            raise RuntimeError('Restart the local receiver with the updated project code.')
        os.environ['CAM_LDS_STORE_URL'] = receiver_url
        os.environ['CAM_LDS_STORE_TOKEN'] = receiver_token
    finally:
        del receiver_token
    print('Local checkpoint receiver connected.')
else:
    print('Manual export mode: results remain on the kernel machine until downloaded.')

# Optional private/gated repository access. Current GGUFs and BitNet are public.
# os.environ['HF_TOKEN'] = getpass('Hugging Face token: ')
''')
md('''
## 4. Prepare GPU workers, including Microsoft BitNet

`prepare()` builds/reuses an isolated Linux environment and launches a subprocess
import check. It never imports torch or llama.cpp into the notebook kernel.
Worker pip is bootstrapped through the kernel's working pip rather than
`ensurepip`, which is unavailable or broken in some Colab Python installations.
An interrupted environment is reused; full bootstrap output is saved in its
`bootstrap.log`. Rerun Section 2, then Section 4 after a helper update; Section 4
reloads the setup helper without restarting the kernel or clearing credentials.
The installed Colab torch is reused if a real GPU operation succeeds. Otherwise
the worker environment gets a CUDA 12.8 torch wheel. llama.cpp is compiled against
the actual compiler-supported GPU architecture rather than forcing CUDA 12.4 on
Blackwell. A CPU llama.cpp fallback remains available if the toolkit cannot target
the allocated card. Worker logs record actual layer placement and fallback.

**Built once per compatible Python/CUDA/GPU combination:** `.cache/envs` and
`.cache/wheels` are reused. **No repeated install when unchanged.** New VMs need
either these compatible Linux artifacts restored or setup again. Large weights are
cached independently of GPU type; H100-to-Blackwell changes do not alter weights.

BitNet uses the official **packed** `microsoft/bitnet-b1.58-2B-4T` checkpoint via
Transformers, with native 1.58-bit-trained weights. Transformers unpacks these for
execution; this is a model-quality experiment, **not a benchmark of the optimized
bitnet.cpp/W2A8 kernels or 1.58-bit runtime memory savings**. This portable path
avoids architecture-specific experimental BitNet GPU extension builds.
''')
code('''
import importlib
import camlds.setup_runtime as setup_runtime
import subprocess

setup_runtime = importlib.reload(setup_runtime)
prepare = setup_runtime.prepare
WORKER_PYTHON = prepare(PROJECT_DIR)
# The lightweight scheduler uses only filelock in the notebook kernel.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'filelock==3.20.0'], check=True)
print('Worker interpreter:', WORKER_PYTHON)
''')
md('''
## 5. Input and context policy

All 990 input rows retain stable one-based IDs. The loader accepts JSON/JSONL and
recovers damaged outer records using the known prompt prefix and ground-truth
field. Embedded log JSON is opaque text. Missing metadata and incomplete ground
truth do not stop prompting when the prompt can be recovered. Ambiguous record
boundaries cause a visible input error rather than silently changing row IDs.

Each source prompt is sent unchanged where it fits. There is **no label cleanup**
and no appended reminder. No separate ground-truth field is sent to the model.

| Model | Context tokens |
|---|---:|
| qwen1b / hermes / llama3b / qwen7b | 16,384 |
| mistral7b | 20,480 |
| bitnet2b | 4,096 |

Generation is greedy (`temperature=0`, no repetition penalty), with a maximum of
**2,048 new tokens** and the model's normal end-of-turn stopping. These are fixed
baseline settings, not tuned in response to answer quality. The finite cap prevents
repetitive output from running indefinitely; text ending at the cap is still saved.

To preserve input, the output allowance can shrink to **256 tokens** for long
queries. If the input still exceeds the context, keep its complete original
instruction header and the leading log text that fits, dropping only the log tail.
This is the selected context-limit policy; it changes the input on those rows.
The CSV records the original prompt, exact effective prompt, truncation flag,
original/input/output token counts, and actual output allowance. There is no
chunking or synthesis of multiple model answers. If even the instructions cannot
fit, record an execution error and continue to the next row.
''')
code('''
from camlds.core import load_queries, OUTPUT_SUBDIR
from camlds.models import MODELS

QUERY_RECORDS = load_queries(PROJECT_DIR/'data/llm_queries.json', expected=990)
assert set(SELECTED_MODELS) <= MODELS.keys()
print('Rows:', len(QUERY_RECORDS))
print('Recovered inputs:', sum(bool(r['_recovery']) for r in QUERY_RECORDS))
print('Original prompts preserved; overlong inputs are shortened only to fit context.')
print('Raw baseline output:', PROJECT_DIR/OUTPUT_SUBDIR)
''')
md('''
## 6. Collect all 990 raw responses per model

This is the full experiment, not a quality test. Every row gets one generation
unless it already has a collected response in this same baseline. All returned
text is saved verbatim in `raw response`, including empty strings, repetitions,
invalid JSON, and partial answers stopped by the token cap. No output is parsed,
scored, corrected, or rejected here.

Each model has a stable CSV and a manifest pinning the dataset, model revision,
and baseline settings. `completed` means the engine returned text, not that the
answer passed any quality check. `execution_error` means no response was returned.

Python generation errors are recorded and the worker continues to later rows.
A native worker crash leaves a row marker; the supervisor starts a new worker,
skips that row for the rest of the pass, and continues. `RETRY_PASSES` applies only
to missing responses/execution errors. Persistent errors remain explicit and can
be retried in another session. GPU/runtime availability cannot be guaranteed.

Transient checkpoint transfers retry automatically without generating another
answer. A persistent transfer failure or local disk error stops with its cause;
responses already saved on Colab remain available for resume. Stop this cell to
terminate workers; rerunning resumes missing IDs without repeating collected
responses. Keep the local receiver and tunnel running. One model at a time is the
default; optional parallel workers share the same GPU resources and may be slower.
Use disjoint model lists if running separate notebooks on the same Colab server.
''')
code('''
from camlds.runner import run_suite, summary, print_report
from camlds.sync import Mirror

if REQUIRE_LOCAL_MIRROR and Mirror.from_env() is None:
    raise RuntimeError('Connect the local checkpoint receiver in Section 3 first.')
run_report = run_suite(PROJECT_DIR, WORKER_PYTHON, selected=SELECTED_MODELS,
                       max_parallel=MAX_PARALLEL_MODELS, passes=RETRY_PASSES)
print_report(run_report)
''')
md('''
## 7. Coverage and execution diagnostics

The target is **990 collected responses per model**, regardless of content.
`retryable_ids` contains only rows without a returned response. The CSV preserves
raw text; transport checks protect row identity and checkpoint consistency only.
Response quality assessment belongs in a separate analysis notebook.

`results/raw_baseline/*.worker.log` records model loading and tracebacks.
`*.attempts.jsonl` records raw responses, token budgets and execution errors.
CSV rows also contain errors and effective prompts. Prior constrained observations
cannot be imported as this baseline because generation settings differed.
''')
code('''
coverage = summary(PROJECT_DIR)
print_report(dict(coverage=coverage, incomplete={}))
# Full list when needed:
# {entry['file']: entry['retryable_ids'] for entry in coverage}
''')
md('''
## 8. Export and reusable local work

With the receiver connected, CSVs and manifests are already in your Windows
project's `results/raw_baseline` folder. In manual-transfer mode, download this
folder using VS Code's Colab explorer. Upload the CSVs **and manifests** to the
same location before resuming on a new VM. Colab storage is ephemeral.

**Prepare locally once:** the dataset, notebook bundle, receiver, tests and analysis.
**Reuse downloaded weights:** `.cache/huggingface` on the same VM, or transfer it to
another VM. **Reuse compatible Linux builds:** `.cache/envs` and `.cache/wheels`;
Windows binaries cannot run on Colab. The old compatible worker environment can
be reused even though output-format enforcement is no longer needed. No runtime
restart is required merely to update these Python helpers.

Microsoft BitNet uses its official packed checkpoint via Transformers; weights
are unpacked for execution. This is not an optimized bitnet.cpp throughput test.
''')
code('''
# Optional manual export after stopping active workers:
# import shutil
# archive_path = shutil.make_archive(str(PROJECT_DIR/'CAM_LDS_raw_baseline'),
#                                    'zip', PROJECT_DIR/'results/raw_baseline')
# print('Download with VS Code Colab explorer:', archive_path)
''')
md('''
## References and verification limits

Local tests cover malformed input recovery, exact raw-text preservation, sparse
resume, context fitting, execution-error continuation and the local receiver.
Actual GPU inference and all 990 outputs must be confirmed on Colab.

- [Microsoft BitNet checkpoint](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Cloudflare tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
''')

for i, cell in enumerate(cells):
    cell['id'] = f'camlds-{i:02}'
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                          language_info=dict(name='python'), colab=dict(name='CAM-LDS resumable GPU evaluation')),
                nbformat=4, nbformat_minor=5)
path = ROOT / '(2)CAM_LDS_GPU_CSV_Checkpoints_9_10.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Built {path.name}: {len(cells)} cells, {path.stat().st_size:,} bytes')
