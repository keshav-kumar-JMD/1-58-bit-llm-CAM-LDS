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
# CAM-LDS: resumable low-bit model evaluation on Colab

Run the attached **990 original CAM-LDS queries** against five GGUF models plus
**Microsoft BitNet b1.58 2B4T**. Ground truth is saved for scoring and never added to
inference messages. The authors' eight label-cleaning substitutions remain in order;
embedded log JSON is opaque text, so incomplete log objects are acceptable.

The workflow is now **settings → portable files → local checkpoint connection →
GPU setup → input check → smoke test → full run → coverage/debugging**.
Reusable code lives in `camlds/`; this notebook embeds an identical copy and the
attached dataset, so VS Code can execute it on a fresh remote Colab kernel without
trying to open a Windows path on Linux. The original notebook, including failure
outputs, is retained in `archive/original_CAM_LDS_9_10.ipynb`.

Each model writes `results/llm_responses_<model>_camlds.csv`, **without a timestamp**.
The first nine columns are preserved; status and diagnostics follow them. Every
source row is represented, but **only `status=ok` is a validated prediction**.
`pending`, `failed`, and invalid imported rows are retried on every resume. No
classification is invented to make coverage appear complete.
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
MAX_PARALLEL_MODELS = 1  # Set 2 on a large GPU after the smoke test; both share VRAM.
RETRY_PASSES = 2        # Each failed chunk also gets three bounded output attempts.
REQUIRE_LOCAL_MIRROR = True  # Recommended for your VS Code -> Colab setup.
RUN_SMOKE_TEST = True
RUN_FULL_SUITE = True
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
print('Portable helpers ready; query SHA256:', hashlib.sha256((PROJECT_DIR/'data/llm_queries.json').read_bytes()).hexdigest())
del PAYLOAD
''', hidden=True)
md('''
## 3. Write checkpoints to this Windows project's local `results` folder

VS Code's Colab kernel runs remotely; `Path.cwd()` is not your Windows workspace.
For automatic local writes, start these in **two LOCAL VS Code terminals**, with
this project as the working directory:

```powershell
python -m camlds.local_store --project .
```

```powershell
cloudflared tunnel --url http://127.0.0.1:8765
```

**Local one-time setup:** install `cloudflared` following its
[official download instructions](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/).
The receiver needs only Python's standard library. An existing HTTPS tunnel you
control works too. Paste the tunnel's HTTPS URL and the receiver's token in the
prompts below. The token is kept in process environment, not notebook source or
saved output. The receiver binds to loopback and accepts authenticated writes only
to known model checkpoint paths. Keep both local terminals running; no Drive mount
is used. The tunnel transports dataset/result content to your local receiver.

Every completed row is first saved on Colab, then acknowledged by the local
receiver. A connection/write failure stops the worker; rerunning reconciles saved
successes. New sessions automatically restore local CSVs and model revision pins.
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
        Mirror(receiver_url, receiver_token).call('/snapshot/qwen1b')
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
## 5. Input recovery and context policy

The input is JSON Lines, not one JSON document. Strict JSON/JSONL parsing is tried
first. Damaged outer records can be recovered using the known prompt prefix and a
`ground_truth` / `ground truth` field, preserving the prompt and raw incomplete truth
value. Missing scenario/variant/step is allowed. Recovery is marked in the CSV.
The expected count stays 990; ambiguous/missing records cause a visible error
instead of silently shifting IDs. No parser can reconstruct absent log text.

| Model key | Quantization | Active context | Native configured limit |
|---|---|---:|---:|
| qwen1b | IQ1_M | 16,384 | 32,768 |
| hermes | IQ1_M | 16,384 | 32,768 |
| bitnet2b | native BitNet b1.58 | 4,096 | 4,096 |
| llama3b | IQ3_M (original comparison model, not 1-bit) | 16,384 | 131,072 |
| mistral7b | IQ1_M | 20,480 | 32,768 |
| qwen7b | IQ1_M | 16,384 | 32,768 |
| qwen1b_q4 (optional control) | Q4_K_M | 16,384 | 32,768 |

Generation uses **768 tokens initially, 1,024 on retries**, with 1,024 reserved
before any call plus a 32-token margin. Workers count the exact tokens sent through
their own chat template and write a per-row token audit. No global 16K override is
applied to BitNet. HF tokenizer measurements are in `review/dataset_token_audit.json`;
the Llama measurement uses the public BitNet Llama-3 vocabulary as a proxy, so the
worker's GGUF audit is authoritative.

For overlong inputs, all cleaned log text is split into contiguous chunks at
newlines where possible, with the original instructions repeated. Each chunk must
produce a valid answer. Ranked IDs are combined by reciprocal-rank fusion (constant
60); confidence is the most attack-confident chunk; the explanation is marked as
an aggregation. This changes the experiment for those rows: filter `chunks > 1`
when comparing single-prompt performance. No ground-truth labels enter aggregation.
''')
code('''
from camlds.core import load_queries, clean_prompt
from camlds.models import MODELS

QUERY_RECORDS = load_queries(PROJECT_DIR/'data/llm_queries.json', expected=990)
assert set(SELECTED_MODELS) <= MODELS.keys()
print('Rows:', len(QUERY_RECORDS))
print('Recovered inputs:', sum(bool(r['_recovery']) for r in QUERY_RECORDS))
print("Prompts changed by authors' cleanup:", sum(clean_prompt(r['prompt']) != r['prompt'] for r in QUERY_RECORDS))
''')
md('''
## 6. Smoke test the saved Qwen failures

Original saved diagnostics: IDs **2 / 89 / 172** had **1,026 / 976 / 2,066 input
tokens**, each generated **2,048 tokens**, and all ended in unterminated JSON.
The outputs repeated `ATT`, `NORMAL`, or digits. These fit the 16K context: increasing
context did not fix the failure. The original grammar allowed unbounded arrays,
arbitrary strings, and unlimited explanations. IQ1_M degradation may contribute,
but the saved evidence alone does not prove quantization is the sole cause.

The fix bounds generation to ten technique IDs, enumerated confidence categories
and a 320-character explanation. GGUF decoding blocks tokens that would complete a
repeated technique ID; validation still rejects duplicate IDs. This constraint is
recorded in raw attempt diagnostics as `distinct-techniques-v1`. Earlier successful
rows remain preserved. A repetition
penalty and two deterministic-seed, low-temperature retries provide another chance.
Complete valid JSON is accepted even if the backend reports the token cap. Grammar
enforces formatting, not classification correctness or actual ATT&CK catalogue membership.

The optional Q4 comparison gets its own model key and CSV, so it cannot replace
or relabel the IQ1_M results. The smoke test saves real results and full raw attempt
diagnostics, then the full run resumes around those IDs.

If a GGUF worker crashes, its log tail and signal name now appear automatically.
GGUF uses an explicit bounded GBNF grammar, avoiding the schema converter's invalid
escaped dot and deeply nested character expansion. The native grammar parser is
checked before inference; a rejected grammar raises a Python error before sampling.
CUDA graphs are disabled in GGUF workers as an A100 compatibility workaround;
GPU layer offload stays enabled. Fresh-worker retries also disable flash attention
and reduce batches. These execution changes preserve saved results and revision pins.
''')
code('''
import importlib
import camlds.runner as runner
runner = importlib.reload(runner)
from camlds.runner import run_suite, summary, print_report

if RUN_SMOKE_TEST:
    smoke = run_suite(PROJECT_DIR, WORKER_PYTHON, selected=['qwen1b', 'bitnet2b'],
                      max_parallel=1, ids=[2, 89, 172], passes=RETRY_PASSES)
    print_report(smoke, ids=[2, 89, 172])
    if smoke['incomplete']:
        raise RuntimeError('Smoke test incomplete; inspect worker/attempt logs before the full dataset.')

# Optional diagnostic comparison (separate CSV):
# run_suite(PROJECT_DIR, WORKER_PYTHON, selected=['qwen1b_q4'], ids=[2,89,172])
''')
md('''
## 7. Full resumable suite and concurrent models

The default loads one model at a time for predictable VRAM use. Set
`MAX_PARALLEL_MODELS=2` to run two worker processes on the **same allocated GPU**.
The scheduler waits for the first model to load, checks current free memory before
admitting another, and retries incomplete runs in fresh worker processes. VRAM
estimates are conservative heuristics; concurrent execution is not guaranteed to
be faster. Model loading tries smaller batches, disables flash attention, reduces
GPU layers, then tries CPU. Worker crashes do not poison the notebook kernel.

You can also open a second notebook attached to the **same Colab server/filesystem**
and choose a disjoint `SELECTED_MODELS` list, using the same `PROJECT_DIR`. Compare
the printed kernel host and project paths. Different models have independent CSVs
and locks; a duplicate model worker is refused. Prefer the single supervisor above
for coordinated VRAM admission. Do not launch the same model on separate VMs.

Pro+ permits increased compute access but does not guarantee multiple managed VMs,
their GPU types, or a specific simultaneous-session quota. Sharing one allocated
GPU through subprocesses does not depend on receiving a second VM. Background
execution is still limited by Colab's runtime lifetime and compute balance.

Stop this cell to terminate its workers safely. On rerun, every non-`ok` ID is tried,
including holes before the largest ID. Two passes are a bounded default; later
sessions can retry remaining failures indefinitely without losing successes.
''')
code('''
if RUN_FULL_SUITE:
    if REQUIRE_LOCAL_MIRROR and Mirror.from_env() is None:
        raise RuntimeError('Connect the local checkpoint receiver first.')
    run_report = run_suite(PROJECT_DIR, WORKER_PYTHON, selected=SELECTED_MODELS,
                           max_parallel=MAX_PARALLEL_MODELS, passes=RETRY_PASSES)
    print_report(run_report)
''')
md('''
## 8. Coverage, debugging, and existing CSV migration

Completion means **990 validated `ok` rows per model**, not 990 attempted rows.
Inspect `retryable_ids` below. In `results/`, `*.worker.log` contains GPU loading and
tracebacks, `*.attempts.jsonl` contains full raw responses, token counts, finish
reasons and validation errors, and `*.token_audit.json` records context plans.
Manifests persist dataset hashes, model revisions, and output policy. Changed inputs
or incompatible settings are refused on resume; GPU/batch fallback can change
without changing source IDs. CSV write errors remain visible and stop the worker.

**Existing Google Drive CSVs:** download a model's old nine-column CSV locally,
then use the optional migration below before its first new run. It checks IDs and
source fields. Responses satisfying the new validator are explicitly marked
`legacy-accepted` and retained; invalid responses remain retryable. This records
mixed historical provenance rather than pretending the old settings were identical.
Never copy several timestamped files on top of one another. No previous result CSVs
were supplied in this project, so none have been migrated automatically.
''')
code('''
coverage = summary(PROJECT_DIR)
for entry in coverage:
    print(entry['file'], '| valid:', entry['valid'], '/990 | missing/failed:', entry['retryable_ids'])

# Run ONCE on a LOCAL Python kernel/terminal before connecting the receiver,
# or on Colab after uploading the old file and before the first model run:
# from camlds.core import import_legacy, load_queries
# import_legacy(Path('old_qwen1b_timestamp.csv'),
#               PROJECT_DIR/'results/llm_responses_qwen1b_camlds.csv', QUERY_RECORDS)

# Read the most recent failed attempts without dumping the whole dataset:
# import json
# entries = [json.loads(line) for line in (PROJECT_DIR/'results/qwen1b.attempts.jsonl').read_text().splitlines()]
# [e for e in entries if e.get('status') != 'ok'][-3:]
''')
md('''
## 9. Export fallback and reusable caches

With the receiver connected, CSVs and manifests are already on your computer.
For manual mode, or to collect remote raw diagnostics, this creates a stable results
archive; use VS Code's Colab server explorer download action. Extract its contents
into your local project's `results/`. Upload the CSVs **and manifests** before a new
manual-mode session. Zip files are snapshots; they do not update themselves.

**Reusable locally:** dataset, source, CSVs, manifests, tests, tokenizer files, and
downloaded weights. **Reusable only on compatible Linux/GPU builds:** wheel caches;
an environment also depends on Python, CUDA and installed torch. Recreating a venv
from a cached wheel is cheaper than compiling it again. Colab VM disks are not
persistent; cache transfer is necessary after VM deletion if no persistent storage
is used. This notebook intentionally does not mount Google Drive.
''')
code('''
# Optional snapshot after stopping active workers:
# import shutil
# archive_path = shutil.make_archive(str(PROJECT_DIR/'CAM_LDS_results'), 'zip', PROJECT_DIR/'results')
# print('Download using VS Code Colab explorer:', archive_path)
''')
md('''
## Review evidence and primary sources

Offline checks cover the actual 990 records, malformed input recovery, exact chunk
coverage, bounded response validation, sparse resume, interrupted writes, changed
dataset/settings rejection, retry diagnostics, ground-truth isolation and locks.
Actual CUDA builds, VRAM/throughput, and all model predictions must still be tested
on your allocated Colab GPU. No claim of 990 successful predictions has been made.

- [CAM-LDS original inference/cleanup](https://github.com/ait-aecid/attack-manifestations-interpretation/blob/44028d8bd40a4a1d8bbbc6ee33261d47cb433827/get_llm_responses.py)
- [Microsoft BitNet checkpoint, 4K limit and inference caveats](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [Transformers BitNet quantization](https://huggingface.co/docs/transformers/main/quantization/bitnet)
- [Official BitNet optimized GPU implementation](https://github.com/microsoft/BitNet/tree/main/gpu)
- [llama-cpp-python installation and CUDA build flags](https://github.com/abetlen/llama-cpp-python#installation)
- [Qwen 1.5B configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)
- [OpenHermes configuration](https://huggingface.co/vicgalle/OpenHermes-Qwen1.5-1.8B/blob/main/config.json)
- [Mistral configuration](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json)
- [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Colab resource and Pro+ limits](https://research.google.com/colaboratory/faq.html)
- [Colab VS Code user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
- [Cloudflare Quick Tunnel setup](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
''')

for i, cell in enumerate(cells):
    cell['id'] = f'camlds-{i:02}'
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                          language_info=dict(name='python'), colab=dict(name='CAM-LDS resumable GPU evaluation')),
                nbformat=4, nbformat_minor=5)
path = ROOT / '(2)CAM_LDS_GPU_CSV_Checkpoints_9_10.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Built {path.name}: {len(cells)} cells, {path.stat().st_size:,} bytes')
