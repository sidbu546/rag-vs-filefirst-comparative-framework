SSFR matched context-difference version

Key additions:
- Explicit answer modes: grounded_context, partial_context, no_context
- Direct visible difference between runs with and without source files
- Inline [Doc X] citations only when usable file evidence is present
- General-knowledge answers without [Doc X] citations when no usable file evidence is present
- New CSV fields:
  context_status
  answer_mode_used
  used_general_knowledge
  retrieved_docs_count
  top_retrieval_score
  avg_retrieval_score
  query_coverage
- Query file exclusion from corpus remains enabled by default
- H200 quantization support remains available: none, 8bit, 4bit


Per-query GPU telemetry update:
- gpu_util_percent now records average GPU utilization sampled during each query.
- gpu_mem_percent now records peak GPU memory percent sampled during each query.
- Added gpu_util_max_percent, gpu_mem_avg_percent, gpu_mem_peak_mb, gpu_mem_torch_peak_mb, and gpu_monitor_samples.
- Throughput, latency, and cost remain logged per query.
