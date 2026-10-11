# CAM-LDS low-bit evaluation

Open [(2)CAM_LDS_GPU_CSV_Checkpoints_9_10.ipynb](./%282%29CAM_LDS_GPU_CSV_Checkpoints_9_10.ipynb) in VS Code and select your Colab kernel. The notebook is self-contained: its bootstrap carries the exact attached dataset and the helper code. Run its numbered sections in order.

The original notebook and saved failures are preserved in `archive/original_CAM_LDS_9_10.ipynb`. The new code is readable in `camlds/`; regenerate the embedded copy after edits with `python tools/build_notebook.py`.

## Local CSVs with a remote Colab kernel

Your VS Code notebook is local, but its Colab Python process cannot access Windows files directly. For automatic writes to **this project's `results/`**, run in a local terminal:

```powershell
python -m camlds.local_store --project .
```

In another local terminal, after installing [cloudflared](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/):

```powershell
cloudflared tunnel --url http://127.0.0.1:8765
```

Enter the HTTPS address and receiver token in notebook section 3. Keep both terminals running. The receiver uses standard-library Python, requires a token for every request, and writes only known model checkpoint paths. The HTTPS tunnel transports the query and result content. No Google Drive folder is used.

If you do not want the receiver, set `REQUIRE_LOCAL_MIRROR=False`. Use VS Code's Colab filesystem download/upload actions to transfer `results/` between sessions. A CSV on `/content` is **not** persistent local Windows storage.

## Results and resume

- One stable `results/llm_responses_<model>_camlds.csv` per model, with all 990 source IDs.
- `status=ok` means a validated prediction; `pending` and `failed` rows do not count as results and are retried, including early gaps.
- CSVs retain the original nine columns, then add status, attempts, error, chunk/token counts, recovery notes and configuration provenance.
- Every update uses a flushed temporary file and atomic replacement. An interrupted update leaves the preceding full CSV.
- Per-model manifests preserve dataset identity, model revision and generation policy across sessions. Keep manifests with CSVs.
- Complete raw attempts and engine logs are retained on the execution machine. Download these for debugging when using Colab.
- Optional migration of original nine-column CSVs is described in section 8. Valid historical rows retain `legacy-accepted` provenance; invalid rows are retried. No old result CSVs were present to migrate here.

Microsoft BitNet uses its official packed `microsoft/bitnet-b1.58-2B-4T` checkpoint through Transformers. This tests native ternary model quality. Transformers unpacks weights and does **not** provide the optimized BitNet kernel's memory/speed guarantees. The original Llama IQ3_M model is also retained and clearly labeled as a comparison outside the 1-bit range. Qwen Q4 is an optional separate control.

## Concurrency and changing GPUs

Set `MAX_PARALLEL_MODELS=2` after the smoke test to run separate model processes on the same Colab GPU. The scheduler admits another worker only after the previous load completes and free memory is sufficient by a conservative estimate. Concurrent inference shares VRAM and may be slower. Two notebooks on the same server may use disjoint model lists; a single supervisor coordinates memory more reliably. Do not run the same model on different servers against one receiver.

Native inference libraries live in separate worker processes, so rebuilding or recovering a worker does not require a notebook-kernel restart. Actual GPU capability and CUDA compiler determine the build cache; memory fallback reduces batches/offload. A Colab-side VM replacement still requires reconnecting and restoring files. Pro+ does not guarantee a specific GPU or number of simultaneous managed VMs. See [Colab's FAQ](https://research.google.com/colaboratory/faq.html).

## Validation and limitations

See [review/REVIEW.md](review/REVIEW.md) for the failure diagnosis and [review/dataset_token_audit.json](review/dataset_token_audit.json) for measurements of all 990 attached prompts. The notebook smoke test targets IDs 2, 89 and 172 before the full suite.

```powershell
python -m unittest discover -s tests -v
python tools/build_notebook.py
```

Tests need `filelock`; all other core tests use Python's standard library. `tools/audit_dataset.py` additionally needs `tokenizers` and `jinja2`. GPU workers install their own dependencies through notebook setup.

Local regression tests and tokenizer measurements are completed. CUDA builds and full model inference have **not** been executed on an actual Colab GPU in this workspace. Success on all 990 rows remains measurable runtime coverage, not a promise or fabricated output.
