# CAM-LDS raw-response baseline

Open `(2)CAM_LDS_GPU_CSV_Checkpoints_9_10.ipynb` in VS Code, connect to Colab, and run Sections 1-6 in order. Section 6 runs all 990 rows per selected model; it has no smoke-test or answer-quality gate.

Each query gets one greedy generation. The original prompt is preserved: no cleanup, added instructions, grammar, format enforcement, duplicate suppression, response parsing, or answer repair. A malformed, repetitive, empty, refused or token-limited answer is still a collected response. Execution errors are logged and remain retryable; collected outputs are never retried for quality.

## Updating from the earlier notebook

1. Reopen the updated notebook and stop any old workers.
2. Restart the local receiver terminal with `python -m camlds.local_store --project .`. It prints a new token; the tunnel can stay running.
3. Run notebook Sections 1-5, entering the tunnel URL and new token in Section 3. Section 4 reuses the compatible worker environment.
4. Run Section 6 to collect the full dataset; Section 7 shows coverage.

The previous constrained notebook/code is preserved in `archive/constrained_v1`. Its CSVs in `results/` remain untouched and do not count as baseline data. This is a new generation policy and requires new observations.

## Local checkpoints and resume

In two local terminals, from this project:

```powershell
python -m camlds.local_store --project .
```

```powershell
.\tools\cloudflared.exe tunnel --url http://127.0.0.1:8765
```

Keep them running. CSVs use stable names in `results/raw_baseline/`, one per model, without timestamps. The first six columns retain source metadata, original prompt and ground truth; `raw response` stores exact returned text. Other columns record execution status, errors, attempts, tokens, finish reason, exact effective prompt, truncation and experiment identity. Ground truth is never added to inference messages.

`completed` means text returned, not quality passed. Empty text is distinguishable from a missing response by status. Reruns skip completed IDs and retry only missing/failed execution. Atomic CSV replacement and pinned manifests support resume across sessions. The receiver protects checkpoint identity without inspecting answer content. In manual mode, transfer both CSVs and manifests between the same `results/raw_baseline` folders.

## Context and generation

Greedy decoding, no repetition penalty, seed 42, up to 2,048 new tokens. Generation stops at the normal model end marker or token cap. Long inputs first reduce the output allowance, down to 256 tokens; then, if necessary, the log tail is shortened while preserving original instructions and the leading log excerpt. A 16-token safety margin is kept. The complete original prompt and actual shortened prompt are recorded. There is one response, no chunk aggregation. Settings are recorded in the per-model manifest.

Microsoft BitNet has 4,096 context tokens; the GGUF models use 16,384 (Mistral: 20,480). BitNet uses the official packed checkpoint through Transformers, which unpacks weights for execution. This is not an optimized bitnet.cpp performance benchmark. Model tokenization and chat templates differ; token budgets and truncation must be considered in the journal analysis.

## Execution reliability and reusable setup

One model at a time is the default. Optional concurrent workers share GPU memory and compute. Python generation errors do not stop later rows; native crashes are isolated to workers and the affected row is skipped until a retry pass. Storage errors remain visible and stop work rather than risk data loss. Persistent hardware/runtime failures cannot be turned into model responses.

**Prepare locally once:** dataset, notebook bundle, receiver and analysis. **Reuse compatible Linux environments and wheels:** `.cache/envs` and `.cache/wheels`. **Reuse weights:** `.cache/huggingface`. Restoring caches avoids downloads/builds on a new VM; Windows binaries are not portable to Colab. Python helper updates do not require restarting the Colab kernel.

```powershell
python -m unittest discover -s tests -v
python tools/build_notebook.py
```

Tests use `filelock`. The optional tokenizer audit uses `tokenizers` and `jinja2`; it writes `review/raw_baseline_token_audit.json`. Historical token measurements reflect the earlier constrained/cleaned experiment and must not be used as baseline measurements. No full Colab GPU experiment has been run from this local workspace.

## Checkpoint recovery update

Temporary receiver/tunnel failures retry the identical saved checkpoint up to five times, with waits of 2, 5, 10 and 20 seconds. This never repeats model generation. The receiver distinguishes temporary file locks from data/configuration conflicts. Persistent transfer failures stop with the HTTP status and receiver reason, preserving Colab checkpoints.

To apply this update to an existing session: reopen the notebook; restart the local receiver terminal (`python -m camlds.local_store --project .`); run Section 2, Section 3 with the new receiver token, then Section 6. Keep the existing tunnel and Colab runtime. No GPU rebuild or model download is needed.
