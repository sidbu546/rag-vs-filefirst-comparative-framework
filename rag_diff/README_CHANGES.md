# RAG changes for visible context/no-context differences

This version makes the system visibly behave differently when usable retrieval context exists versus when it does not.

## New behavior
- `grounded_context`: answer is grounded in retrieved docs and uses inline citations like `[Doc 1]`.
- `partial_context`: answer mixes retrieved evidence with model knowledge; context-derived claims may include `[Doc X]`.
- `no_context`: answer uses model knowledge only and intentionally includes no `[Doc X]` citations.

## New output columns
- `context_status`
- `answer_mode_used`
- `used_general_knowledge`
- `retrieved_docs_count`
- `top_retrieval_score`
- `avg_retrieval_score`
- `query_coverage`

## Important
- `--query_file` is excluded from indexing automatically, so your prompt file is not retrieved as evidence.
- This keeps answers different when you add a real data file versus when you remove it.


Per-query GPU telemetry update:
- gpu_util_percent now records average GPU utilization sampled during each query.
- gpu_mem_percent now records peak GPU memory percent sampled during each query.
- Added gpu_util_max_percent, gpu_mem_avg_percent, gpu_mem_peak_mb, gpu_mem_torch_peak_mb, and gpu_monitor_samples.
- Throughput, latency, and cost remain logged per query.
