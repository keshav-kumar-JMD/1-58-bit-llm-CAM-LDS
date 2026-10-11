# Raw-baseline revision

The earlier implementation incorrectly treated low-quality output as a failed inference. It constrained JSON, rejected repeated technique IDs, retried answers, appended instructions, cleaned source prompts, and aggregated chunks. Those behaviors have been removed from the active experiment. Historical code, notebook and review are preserved under `archive/constrained_v1`.

The current baseline records any returned string without parsing it. Greedy decoding and a finite output cap are fixed, explicit experimental settings. Input shortening is used only to respect context limits, with original and effective prompts saved. No ground-truth field is passed to a backend.

Outputs use `results/raw_baseline`, preserving earlier CSVs without mixing methods. A completed row is never re-run on content grounds. Execution errors and missing outputs remain separately identifiable. A native crash quarantines the running row for the current pass so later rows can execute.

Local tests verify raw output preservation including empty/malformed text, recovery of source records, instruction-preserving context fitting, sparse resume, atomic writes, receiver identity checks and failure continuation. They do not establish prediction quality or successful inference on the remote Colab GPU.

`dataset_token_audit.json` is historical (cleaned/constrained prompts). Use `raw_baseline_token_audit.json` for the new tokenizer-only audit, and per-row runtime token counts as authoritative for actual model inputs.
