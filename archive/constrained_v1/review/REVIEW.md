# Notebook review — 10 October 2026

## Observed Qwen failure

Saved diagnostic output in the original notebook records:

| Source ID | Input tokens | Generated tokens | Finish | Output problem |
|---|---:|---:|---|---|
| 2 | 1,026 | 2,048 | length | Repeated `ATT`/`NORMAL`, repetitive explanation, unterminated string |
| 89 | 976 | 2,048 | length | Repeated `ATT`, unterminated string |
| 172 | 2,066 | 2,048 | length | Repeated digits in a technique string, unterminated string |

All three fit the configured 16,384-token context. More input context therefore does not address the demonstrated failure. The old grammar constrains JSON shape but allows unlimited technique strings, array entries and explanation length. The original runner rejects all `finish_reason=length` responses and has no output retry. Its resume logic starts after the highest saved ID, permanently overlooking earlier missing IDs. An added Q4 test cell calls its benchmark helper before that helper is defined.

The new policy bounds technique syntax/count, confidence values and explanation length; uses repetition penalties and up to three attempts per chunk; validates the actual JSON instead of rejecting the finish reason alone; and durably records every raw attempt. Severe IQ1_M quantization may impair reasoning/output quality, but the saved examples alone do not establish it as the sole cause. The optional Q4 control remains a separate experiment. The proposed fix needs the supplied GPU smoke test to measure its effectiveness.

## Input audit

The attached file is 6,885,747 bytes of JSON Lines with **990** records. Every source record currently parses correctly; malformed JSON inside a prompt's log excerpt is simply text. Raw prompt lengths range from 1,311 to 30,056 characters; the median is 4,236 and the 95th percentile is 18,032. The original eight regex cleanup operations are preserved. Recovery code also supports incomplete outer JSON, literal newlines, missing optional metadata, alternate ground-truth key spelling, and incomplete ground-truth values. Ambiguous input/count mismatches stop visibly instead of renumbering or skipping rows.

The supplied dataset is copied unchanged to `data/llm_queries.json` and embedded in the notebook. Its ground truth is never appended to model input. Existing cleanup removes annotated labels using the authors' rules; any other technique IDs already present in log text remain, as in the original notebook.

## Context audit and experiment changes

All measurements use cleaned prompts and the corresponding HF chat template. Actual workers also count their own formatted tokens before generation. See the machine-readable audit for exact maxima and chunk counts. Qwen's audit reproduces the saved input counts of 1,026, 976 and 2,066, providing a useful cross-check.

| Model | Active context | Maximum measured original formatted input | Policy |
|---|---:|---:|---|
| Qwen 1.5B | 16,384 | 13,974 | Whole prompt |
| OpenHermes Qwen1.5 1.8B | 16,384 | 13,963 | Whole prompt |
| Microsoft BitNet 2B | 4,096 | 10,923 | 234 rows chunked, at most four chunks |
| Llama 3.2 3B | 16,384 | 10,923 (public Llama-3 vocabulary proxy) | Verify actual GGUF template at runtime |
| Mistral 7B v0.3 | 20,480 | 15,561 | Whole prompt; original 16K would chunk five rows |
| Qwen 7B | 16,384 | 13,974 | Whole prompt |

The revised output-format reminder adds roughly 50–60 input tokens. Planning reserves 1,024 output tokens plus 32 safety tokens; generation starts at 768 and retries at 1,024. BitNet stays within its native 4K limit. Contiguous chunking covers all cleaned log text, repeats task instructions and combines the chunk predictions using declared reciprocal-rank fusion. Chunk confidence selects the most attack-confident chunk. The explanation is explicitly marked as an aggregation. These are methodological changes from the original single-call experiment; the CSV's `chunks` field identifies affected rows. Chunking does not preserve cross-chunk interactions, so compare chunked and unchunked results separately.

The old Llama IQ3_M entry is not a 1-bit model; it remains as the user's original comparison. Microsoft BitNet is trained with native ternary weights; the other IQ1_M models are post-training quantizations. Their numerical formats are not identical.

## Persistence, setup and verification

One atomic CSV snapshot per model contains all source IDs and explicit status. All missing/failed IDs are retried on resume, including gaps before the highest success. Valid historical nine-column rows can be explicitly migrated with provenance retained. Full diagnostics and configuration manifests replace silent skipping and in-memory-only revision pins.

Separate Linux worker environments are keyed by Python/CUDA/GPU capability and host torch version. GPU operations are probed before running models; CUDA binaries are compiled for the actual supported architecture, with CPU fallback if the compiler cannot target it. Model loading has conservative memory fallbacks. Native library rebuilds do not require restarting the notebook process. Colab itself can still delete or replace its VM.

The local checkpoint receiver addresses the distinction between a local VS Code notebook and remote Colab execution. It writes into this Windows project's `results/` using authenticated HTTPS-tunneled requests. Its URL/token must be configured by the user; no tunnel has been launched or Colab account changed during this review. Without it, the documented export/import workflow is necessary for cross-VM persistence.

Tests exercise real 990-row loading, malformed-input recovery, source-ID stability, chunk coverage and token budgets, sparse resume, interrupted replacement, changed-input/config rejection, bounded-output retries, accepting valid cap-terminated JSON, ground-truth isolation, and model locks. Notebook JSON/schema and all code-cell syntax are validated. Actual GPU installation, CUDA compatibility, multi-model VRAM usage and classification accuracy remain to be verified by the notebook smoke test/full run.

Primary references: [Microsoft model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), [Transformers BitNet implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/bitnet.py), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [Colab limits](https://research.google.com/colaboratory/faq.html), [Colab VS Code guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide), and [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
