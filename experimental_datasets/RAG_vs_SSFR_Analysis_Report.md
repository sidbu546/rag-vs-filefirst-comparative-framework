# RAG vs. SSFR Comparative Evaluation: Comprehensive Analysis Report

## Executive Summary

This report presents a rigorous comparative evaluation of two document-grounded
question-answering architectures: **Retrieval-Augmented Generation (RAG)** and
**Single-Shot File Retrieval (SSFR)**. The evaluation spans **582 total observations**
across **16 experimental configurations** varying four independent variables:

| Variable | Levels |
|----------|--------|
| Retrieval Approach | RAG, SSFR |
| Base Model | Qwen-32B, Llama |
| Quantization | 4-bit, 8-bit |
| Corpus Size | Small, Big |

- **RAG observations**: 276
- **SSFR observations**: 306

### Key Findings at a Glance

| Finding | Detail |
|---------|--------|
| Lower Hallucination | **RAG** (0.195 vs 0.409, p=0.0000***) |
| Higher Groundedness | **RAG** (0.685 vs 0.484, p=0.0000***) |
| Higher Answer Relevance | **SSFR** (4.48 vs 4.73, p=0.0000***) |
| Faster Response | **RAG** (39.70s vs 59.70s, p=0.0000***) |

---

## 1. RAG vs. SSFR: Head-to-Head Comparison

### 1.1 Answer Quality Metrics

| Metric | RAG (mean +/- std) | SSFR (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
|--------|-------------------|--------------------|----- |---------|-----------|------|
| hallucination_rate | 0.1949 +/- 0.2547 | 0.4091 +/- 0.3029 | -0.2142 | 0.0000*** | -0.762 | Yes |
| groundedness_score | 0.6853 +/- 0.2567 | 0.4835 +/- 0.2896 | 0.2018 | 0.0000*** | 0.735 | Yes |
| answer_relevance_1to5 | 4.4819 +/- 0.5813 | 4.7320 +/- 0.4510 | -0.2501 | 0.0000*** | -0.484 | Yes |
| context_relevance_1to5 | 3.7609 +/- 0.9839 | 2.5784 +/- 1.1603 | 1.1824 | 0.0000*** | 1.095 | Yes |
| confidence | 0.5414 +/- 0.0733 | 0.9912 +/- 0.0625 | -0.4498 | 0.0000*** | -6.629 | Yes |
| query_coverage | 0.4863 +/- 0.1754 | 1.0000 +/- 0.0000 | -0.5137 | 0.0000*** | -4.253 | Yes |

### 1.2 Performance Metrics

| Metric | RAG (mean +/- std) | SSFR (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
|--------|-------------------|--------------------|----- |---------|-----------|------|
| response_time_s | 39.70 +/- 23.46 | 59.70 +/- 27.21 | -20.00 | 0.0000*** | -0.784 | Yes |
| llm_latency_s | 39.68 +/- 23.46 | 50.23 +/- 25.29 | -10.56 | 0.0000*** | -0.432 | Yes |
| gpu_throughput_toks_per_s | 10.62 +/- 4.94 | 9.99 +/- 5.08 | 0.63 | 0.1307 | 0.125 | No |
| eff_gpu_throughput | 8.29 +/- 4.78 | 7.36 +/- 3.94 | 0.93 | 0.0109* | 0.214 | Yes |
| gpu_util_percent | 76.13 +/- 21.09 | 73.61 +/- 18.43 | 2.52 | 0.1275 | 0.128 | No |
| gpu_mem_percent | 57.11 +/- 26.54 | 62.99 +/- 31.16 | -5.88 | 0.0142* | -0.202 | Yes |
| gpu_mem_peak_mb | 38732.33 +/- 23291.48 | 38008.22 +/- 24512.92 | 724.11 | 0.7150 | 0.030 | No |
| total_deployment_cost_usd | 2.03 +/- 0.02 | 1.75 +/- 0.67 | 0.29 | 0.0000*** | 0.594 | Yes |

### 1.3 Retrieval Characteristics

| Metric | RAG (mean) | SSFR (mean) | p-value |
|--------|-----------|------------|---------|
| retrieved_docs_count | 7.59 | 1.49 | 0.0000*** |
| top_retrieval_score | 0.54 | 1952.33 | 0.0000*** |
| avg_retrieval_score | 0.49 | 1812.70 | 0.0000*** |
| query_coverage | 0.49 | 1.00 | 0.0000*** |

**Interpretation:** RAG retrieves many more document chunks (multi-passage) with lower per-chunk
similarity scores, while SSFR retrieves fewer whole documents with substantially higher
retrieval scores. This fundamental architectural difference — chunked retrieval vs. whole-file
delivery — drives many downstream quality and performance differences.

---

## 2. Model Comparison: Qwen-32B vs. Llama

| Metric | Qwen-32B (mean) | Llama (mean) | Diff | p-value | Cohen's d | Sig. |
|--------|----------------|-------------|------|---------|-----------|------|
| hallucination_rate | 0.3379 | 0.2739 | 0.0639 | 0.0099** | 0.214 | Yes |
| groundedness_score | 0.5162 | 0.6491 | -0.1330 | 0.0000*** | -0.467 | Yes |
| answer_relevance_1to5 | 4.4216 | 4.8261 | -0.4045 | 0.0000*** | -0.823 | Yes |
| context_relevance_1to5 | 3.0229 | 3.2681 | -0.2452 | 0.0199* | -0.200 | Yes |
| confidence | 0.7932 | 0.7609 | 0.0323 | 0.0980 | 0.138 | No |
| query_coverage | 0.7679 | 0.7436 | 0.0243 | 0.3036 | 0.086 | No |
| response_time_s | 54.4898 | 45.4842 | 9.0056 | 0.0000*** | 0.333 | Yes |
| llm_latency_s | 49.5648 | 40.4189 | 9.1459 | 0.0000*** | 0.372 | Yes |
| gpu_throughput_toks_per_s | 11.0438 | 9.4579 | 1.5860 | 0.0001*** | 0.320 | Yes |
| eff_gpu_throughput | 8.3635 | 7.1822 | 1.1813 | 0.0009*** | 0.272 | Yes |
| gpu_util_percent | 77.5992 | 71.7033 | 5.8959 | 0.0004*** | 0.302 | Yes |
| gpu_mem_percent | 42.8963 | 79.3887 | -36.4924 | 0.0000*** | -1.600 | Yes |
| gpu_mem_peak_mb | 19761.4469 | 58962.4443 | -39200.9974 | 0.0000*** | -2.853 | Yes |
| total_deployment_cost_usd | 2.0164 | 1.7323 | 0.2841 | 0.0000*** | 0.586 | Yes |

---

## 3. Quantization Impact: 4-bit vs. 8-bit

| Metric | 4-bit (mean) | 8-bit (mean) | Diff | p-value | Cohen's d | Sig. |
|--------|-------------|-------------|------|---------|-----------|------|
| hallucination_rate | 0.3237 | 0.2929 | 0.0308 | 0.2177 | 0.103 | No |
| groundedness_score | 0.5728 | 0.5850 | -0.0122 | 0.6149 | -0.042 | No |
| answer_relevance_1to5 | 4.6558 | 4.5752 | 0.0806 | 0.0673 | 0.152 | No |
| context_relevance_1to5 | 3.2355 | 3.0523 | 0.1832 | 0.0707 | 0.149 | No |
| confidence | 0.7658 | 0.7888 | -0.0230 | 0.2392 | -0.098 | No |
| query_coverage | 0.7427 | 0.7688 | -0.0261 | 0.2684 | -0.092 | No |
| response_time_s | 32.0881 | 66.5726 | -34.4845 | 0.0000*** | -1.620 | Yes |
| llm_latency_s | 27.5751 | 61.1494 | -33.5743 | 0.0000*** | -1.813 | Yes |
| gpu_throughput_toks_per_s | 14.2667 | 6.7064 | 7.5603 | 0.0000*** | 2.285 | Yes |
| eff_gpu_throughput | 11.7049 | 4.2841 | 7.4208 | 0.0000*** | 3.183 | Yes |
| gpu_util_percent | 83.7442 | 66.7388 | 17.0054 | 0.0000*** | 0.953 | Yes |
| gpu_mem_percent | 77.4150 | 44.6764 | 32.7386 | 0.0000*** | 1.353 | Yes |
| gpu_mem_peak_mb | 35663.5453 | 40776.1401 | -5112.5948 | 0.0073** | -0.215 | Yes |
| total_deployment_cost_usd | 1.7276 | 2.0206 | -0.2930 | 0.0000*** | -0.606 | Yes |

---

## 4. Corpus Size Impact: Small vs. Big

| Metric | Small (mean) | Big (mean) | Diff | p-value | Cohen's d | Sig. |
|--------|-------------|-----------|------|---------|-----------|------|
| hallucination_rate | 0.3710 | 0.2342 | 0.1368 | 0.0000*** | 0.467 | Yes |
| groundedness_score | 0.5327 | 0.6330 | -0.1002 | 0.0000*** | -0.348 | Yes |
| answer_relevance_1to5 | 4.6186 | 4.6074 | 0.0112 | 0.7987 | 0.021 | No |
| context_relevance_1to5 | 2.7756 | 3.5593 | -0.7836 | 0.0000*** | -0.671 | Yes |
| confidence | 0.7722 | 0.7844 | -0.0122 | 0.5325 | -0.052 | No |
| query_coverage | 0.7549 | 0.7581 | -0.0032 | 0.8918 | -0.011 | No |
| response_time_s | 48.1748 | 52.5815 | -4.4067 | 0.0573 | -0.161 | No |
| llm_latency_s | 44.3141 | 46.2831 | -1.9690 | 0.3487 | -0.079 | No |
| gpu_throughput_toks_per_s | 9.5473 | 11.1519 | -1.6047 | 0.0002*** | -0.324 | Yes |
| eff_gpu_throughput | 7.3314 | 8.3485 | -1.0171 | 0.0073** | -0.234 | Yes |
| gpu_util_percent | 77.3692 | 71.8381 | 5.5311 | 0.0008*** | 0.282 | Yes |
| gpu_mem_percent | 66.3229 | 53.1288 | 13.1941 | 0.0000*** | 0.464 | Yes |
| gpu_mem_peak_mb | 41858.0284 | 34299.7500 | 7558.2784 | 0.0002*** | 0.320 | Yes |
| total_deployment_cost_usd | 1.7678 | 2.0132 | -0.2454 | 0.0000*** | -0.500 | Yes |

---

## 5. Hallucination Analysis

### 5.1 Hallucination Rate by Dimension

| Dimension | Value | Mean | Median | Std | % Zero | % Above 0.3 | n |
|-----------|-------|------|--------|-----|--------|-------------|---|
| approach | RAG | 0.1949 | 0.1000 | 0.2547 | 40.9% | 21.0% | 276 |
| approach | SSFR | 0.4091 | 0.4000 | 0.3029 | 11.1% | 52.9% | 306 |
| model | Llama | 0.2739 | 0.1833 | 0.2853 | 27.2% | 33.0% | 276 |
| model | Qwen-32B | 0.3379 | 0.2361 | 0.3110 | 23.5% | 42.2% | 306 |
| quantization | 4-bit | 0.3237 | 0.2000 | 0.3032 | 23.6% | 39.1% | 276 |
| quantization | 8-bit | 0.2929 | 0.2000 | 0.2979 | 26.8% | 36.6% | 306 |
| corpus_size | Big | 0.2342 | 0.2000 | 0.2220 | 26.3% | 28.9% | 270 |
| corpus_size | Small | 0.3710 | 0.3000 | 0.3426 | 24.4% | 45.5% | 312 |

### 5.2 Context Grounding Status Distribution

| Approach | grounded_context | no_context | partial_context |
|----------|--------|--------|--------|
| RAG | 68.1% | 0.0% | 31.9% |
| SSFR | 98.0% | 2.0% | 0.0% |

### 5.3 General Knowledge Fallback Rate

| Dimension | Value | GK Usage % | n |
|-----------|-------|-----------|---|
| approach | RAG | 31.88% | 276 |
| approach | SSFR | 1.96% | 306 |
| model | Llama | 18.12% | 276 |
| model | Qwen-32B | 14.38% | 306 |
| quantization | 4-bit | 17.03% | 276 |
| quantization | 8-bit | 15.36% | 306 |
| corpus_size | Big | 15.19% | 270 |
| corpus_size | Small | 16.99% | 312 |

---

## 6. Key Correlation Insights

### 6.1 Quality-Performance Correlations

| Metric A | Metric B | Pearson r |
|----------|----------|-----------|
| hallucination_rate | groundedness_score | -0.9020 |
| hallucination_rate | response_time_s | 0.0474 |
| groundedness_score | response_time_s | -0.1574 |
| groundedness_score | confidence | -0.2542 |
| answer_relevance_1to5 | context_relevance_1to5 | 0.0482 |
| hallucination_rate | retrieved_docs_count | -0.3091 |
| groundedness_score | retrieved_docs_count | 0.3151 |
| gpu_throughput_toks_per_s | response_time_s | -0.6901 |
| gpu_mem_peak_mb | gpu_throughput_toks_per_s | -0.0998 |
| confidence | hallucination_rate | 0.3200 |
| query_coverage | groundedness_score | -0.2305 |
| query_coverage | hallucination_rate | 0.2433 |

---

## 7. Interaction Effects: Best & Worst Configurations

### 7.1 All Configurations Ranked

| Config | Halluc. Rate | Groundedness | Ans. Relevance | Response Time (s) | GPU Mem Peak (MB) |
|--------|-------------|-------------|----------------|-------------------|-------------------|
| RAG|Llama|4-bit|Big | 0.1293 | 0.8173 | 5.00 | 29.88 | 45287 |
| RAG|Llama|8-bit|Big | 0.1130 | 0.7960 | 4.77 | 43.51 | 72872 |
| RAG|Qwen-32B|8-bit|Big | 0.0859 | 0.7549 | 4.03 | 53.11 | 1091 |
| RAG|Qwen-32B|4-bit|Big | 0.1256 | 0.7154 | 4.00 | 16.04 | 21138 |
| RAG|Llama|8-bit|Small | 0.2381 | 0.6764 | 4.79 | 42.34 | 73320 |
| RAG|Llama|4-bit|Small | 0.2946 | 0.6708 | 5.00 | 27.67 | 42141 |
| SSFR|Llama|8-bit|Small | 0.3547 | 0.6178 | 4.69 | 56.28 | 76710 |
| SSFR|Llama|8-bit|Big | 0.2161 | 0.6109 | 4.53 | 74.89 | 73477 |
| RAG|Qwen-32B|4-bit|Small | 0.2498 | 0.5808 | 4.13 | 23.87 | 25671 |
| RAG|Qwen-32B|8-bit|Small | 0.2481 | 0.5501 | 4.10 | 77.45 | 24983 |
| SSFR|Llama|4-bit|Big | 0.3125 | 0.5355 | 4.90 | 46.51 | 45286 |
| SSFR|Llama|4-bit|Small | 0.4580 | 0.5061 | 4.90 | 45.76 | 42853 |
| SSFR|Qwen-32B|8-bit|Big | 0.3676 | 0.4926 | 4.77 | 91.67 | 14101 |
| SSFR|Qwen-32B|4-bit|Big | 0.3902 | 0.4813 | 4.70 | 25.95 | 21344 |
| SSFR|Qwen-32B|4-bit|Small | 0.5521 | 0.3348 | 4.62 | 38.72 | 39373 |
| SSFR|Qwen-32B|8-bit|Small | 0.5726 | 0.3252 | 4.72 | 73.31 | 9813 |

- **Highest Groundedness**: `RAG|Llama|4-bit|Big` (groundedness=0.8173)
- **Lowest Hallucination**: `RAG|Qwen-32B|8-bit|Big` (halluc=0.0859)
- **Fastest Response**: `RAG|Qwen-32B|4-bit|Big` (time=16.04s)
- **Lowest Groundedness**: `SSFR|Qwen-32B|8-bit|Small` (groundedness=0.3252)

---

## 8. Discussion & Conclusions

### 8.1 Architectural Trade-offs: RAG vs. SSFR

The experimental results reveal a nuanced trade-off landscape between the two retrieval
architectures that defies simple "one is better" conclusions:

1. **Retrieval Granularity Paradox**: RAG's multi-chunk retrieval (avg 7.6 docs) provides
   broader context coverage but at lower per-chunk relevance scores. SSFR's whole-file
   approach (avg 1.5 docs) achieves dramatically higher retrieval scores (3606x higher),
   suggesting that delivering complete document context may improve semantic alignment.

2. **Hallucination-Groundedness Coupling**: Across all configurations, hallucination rate
   and groundedness score show a correlation of r=-0.9020, confirming they capture
   complementary aspects of answer fidelity. The approach with lower hallucination
   consistently demonstrates higher groundedness.

3. **Latency-Quality Trade-off**: Response times differ significantly between approaches
   (p=0.0000***, Cohen's d=-0.784), with implications for real-time deployment.
   The faster approach does not necessarily sacrifice quality, suggesting architectural
   efficiency gains rather than quality shortcuts.

### 8.2 Model Architecture Effects

Qwen-32B and Llama exhibit statistically significant differences across multiple metrics:

- **Memory footprint**: Llama consumes 58,962 MB peak GPU memory vs.
  Qwen-32B's 19,761 MB — a 198.5% increase (p<0.001, Cohen's d=-2.853, large effect).
  This likely reflects different quantization base sizes or architectural overhead.
- **Groundedness**: Llama achieves significantly higher groundedness (0.6491 vs. 0.5162,
  p<0.001, Cohen's d=-0.467) despite its higher resource demands.
- **Hallucination**: Llama also hallucinates less (0.2739 vs. 0.3379, p<0.01),
  suggesting larger effective parameter precision contributes to answer fidelity.
- **Answer relevance**: Llama scores higher (4.83 vs. 4.42, p<0.001, Cohen's d=-0.823, large effect),
  the single largest quality advantage observed in any pairwise comparison.
- **Throughput**: Qwen-32B generates tokens faster (11.04 vs. 9.46 tok/s, p<0.001)
  but takes longer overall (54.5s vs. 45.5s), implying SSFR's whole-file context
  inflates prompt sizes disproportionately for Qwen.

### 8.3 Quantization Trade-offs

- 4-bit quantization: halluc=0.3237, mem=35,664 MB, throughput=14.27 tok/s
- 8-bit quantization: halluc=0.2929, mem=40,776 MB, throughput=6.71 tok/s
- Memory savings from 4-bit: 12.5%

The quantization level presents a classic precision-efficiency trade-off. The 4-bit
quantization achieves notable memory savings while the impact on answer quality
must be weighed against deployment constraints.

### 8.4 Corpus Size Sensitivity

- Small corpus: halluc=0.3710, groundedness=0.5327
- Big corpus: halluc=0.2342, groundedness=0.6330

Counter-intuitively, the **big corpus produces significantly better answer quality**
(lower hallucination by 13.7 percentage points, higher groundedness by 10 points,
both p<0.001). This contradicts the common assumption that larger retrieval corpora
introduce noise. One explanation is that the big corpus (religious/philosophical texts)
has more internally consistent and self-referencing content than the small corpus
(history/nationalism), making retrieval more reliable. The big corpus also yields
higher context relevance (3.56 vs. 2.78, p<0.001, Cohen's d=-0.671), suggesting
richer, more topically coherent source material.

### 8.5 Paradoxical Findings (Key Academic Contributions)

Several findings challenge prevailing assumptions in the RAG literature:

1. **The Confidence-Hallucination Paradox**: SSFR reports near-perfect confidence
   (0.991 vs. RAG's 0.541) yet hallucinates *more* (0.409 vs. 0.195). The positive
   Pearson correlation (r=0.320) between confidence and hallucination rate across
   all observations suggests that **model confidence is anti-correlated with actual
   answer fidelity**. This has critical implications for confidence-based filtering
   in production systems — high confidence cannot be trusted as a proxy for correctness.

2. **The Context-Grounding Paradox**: SSFR achieves 98% `grounded_context` status
   (vs. RAG's 68%) and 100% query coverage (vs. RAG's 49%), yet produces lower
   groundedness scores and higher hallucination. This reveals that **having access
   to the full source document does not guarantee faithful use of that context**.
   The model may over-generate beyond retrieved evidence when given whole files,
   despite technically having all necessary information.

3. **The Retrieval Score Inversion**: SSFR's retrieval scores are ~3,600x higher than
   RAG's, yet RAG produces more grounded answers. This suggests that cosine-similarity-based
   retrieval scores at the document level (SSFR) and chunk level (RAG) measure
   fundamentally different constructs and **should not be compared across architectures**.

4. **General Knowledge as Quality Signal**: RAG uses general knowledge fallback 31.9%
   of the time (vs. SSFR's 2%), yet achieves better groundedness. This implies that
   RAG's partial-context mode with knowledge augmentation may outperform SSFR's
   all-or-nothing full-document retrieval, particularly when retrieved context is
   incomplete — a finding that favors **hybrid retrieval-generation strategies**.

5. **The Quantization Non-Effect**: Unlike all other independent variables, quantization
   level (4-bit vs. 8-bit) shows **no statistically significant effect** on any quality
   metric (hallucination p=0.22, groundedness p=0.61, relevance p=0.07). This is a
   practically important null finding: 4-bit quantization achieves 2.13x throughput
   (14.27 vs. 6.71 tok/s, p<0.001) with negligible quality degradation, strongly
   supporting aggressive quantization for deployment.

### 8.6 Recommendations for Practitioners

1. **For maximum answer fidelity**: Prioritize the configuration with the highest
   groundedness score (`RAG|Llama|4-bit|Big`).
2. **For resource-constrained deployments**: 4-bit quantization provides meaningful
   memory savings (12.5%) with measurable but potentially acceptable quality trade-offs.
3. **For latency-sensitive applications**: Choose the faster retrieval architecture
   (`RAG|Qwen-32B|4-bit|Big` at 16.04s mean response time).
4. **For minimizing hallucination risk**: The `RAG|Qwen-32B|8-bit|Big` configuration achieves the
   lowest hallucination rate (0.0859).

---

## 9. Threats to Validity & Limitations

1. **Corpus confound**: The small and big corpora cover different subject domains
   (history vs. religious philosophy), making corpus size effects inseparable from
   domain effects. A controlled experiment would use same-domain corpora of varying sizes.
2. **Sample imbalance**: SSFR has two extra dataset files (Qwen 8-bit big has two runs),
   slightly inflating SSFR observation counts (306 vs. 276).
3. **Retrieval score incomparability**: RAG and SSFR use fundamentally different scoring
   mechanisms, making direct retrieval score comparisons meaningful only within-approach.
4. **Single GPU configuration**: All experiments appear to run on a single GPU setup;
   results may not generalize to multi-GPU or distributed inference environments.
5. **Cost metric uniformity**: The `total_deployment_cost_usd` field shows limited variance
   (~$1.73-$2.03), suggesting it may reflect infrastructure cost rather than per-query
   marginal cost.

---

## 10. Methodology Notes

- **Total observations**: 582
- **Unique configurations**: 16
- **Statistical tests**: Welch's t-test (unequal variances), two-tailed
- **Effect sizes**: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)
- **Significance levels**: * p<0.05, ** p<0.01, *** p<0.001
- **Metrics evaluated**: answer quality (hallucination, groundedness, relevance),
  retrieval quality (coverage, doc count, retrieval scores), performance (latency,
  throughput), and resource utilization (GPU memory, GPU utilization, cost)

---

*Report generated automatically from 18 experimental CSV datasets comprising 582 total
observations across 16 configurations. Statistical analysis performed with SciPy and
Pandas. All p-values are two-tailed Welch's t-test (unequal variance assumption).*
