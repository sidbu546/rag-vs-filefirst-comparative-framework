from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from gpu_metrics import QueryGpuMonitor
from rag_core import rag_advanced
from rag_judges import judge_claims, judge_relevance


def usd_cost_from_time(
    gpu_seconds: float,
    cpu_seconds: float = 0.0,
    gpu_cost_per_hour: float = 0.0,
    cpu_cost_per_hour: float = 0.0,
    fixed_cost_usd: float = 0.0,
) -> float:
    gpu_cost = (gpu_seconds / 3600.0) * gpu_cost_per_hour
    cpu_cost = (cpu_seconds / 3600.0) * cpu_cost_per_hour
    return float(gpu_cost + cpu_cost + fixed_cost_usd)


@dataclass
class RagEvalRow:
    query: str
    answer: str
    hallucination_rate: float
    groundedness_score: float
    answer_relevance_1to5: int
    context_relevance_1to5: int
    confidence: float
    response_time_s: float
    llm_latency_s: float
    gpu_throughput_toks_per_s: float
    eff_gpu_throughput: float
    gpu_util_percent: Optional[float]
    gpu_mem_percent: Optional[float]
    gpu_util_max_percent: Optional[float]
    gpu_mem_avg_percent: Optional[float]
    gpu_mem_peak_mb: Optional[float]
    gpu_mem_torch_peak_mb: Optional[float]
    gpu_monitor_samples: int
    total_deployment_cost_usd: float
    top_source: str
    context_status: str
    answer_mode_used: str
    used_general_knowledge: bool
    retrieved_docs_count: int
    top_retrieval_score: float
    avg_retrieval_score: float
    query_coverage: float


def evaluate_rag(
    queries: List[Dict[str, Any]],
    retriever,
    rag_llm,
    judge_llm,
    top_k: int = 3,
    min_score: float = 0.0,
    max_claims: int = 12,
    gpu_index: int = 0,
    gpu_cost_per_hour: float = 0.0,
    cpu_cost_per_hour: float = 0.0,
    fixed_cost_usd: float = 0.0,
) -> Tuple[List[RagEvalRow], List[Dict[str, Any]]]:
    rows: List[RagEvalRow] = []
    detailed: List[Dict[str, Any]] = []

    for item in queries:
        q = item["query"]

        monitor = QueryGpuMonitor(gpu_index=gpu_index).start()
        try:
            result = rag_advanced(
                q, retriever, rag_llm,
                top_k=top_k, min_score=min_score,
                return_context=True
            )
        finally:
            gpu_stats = monitor.stop()

        answer = result.get("answer", "")
        context = result.get("context", "")
        confidence = float(result.get("confidence", 0.0))
        sources = result.get("sources", []) or []
        top_source = sources[0]["source"] if sources else ""

        timings = result.get("timings", {}) or {}
        response_time_s = float(timings.get("total_s", 0.0))
        llm_latency_s = float(timings.get("llm_s", 0.0))
        gpu_tps = float(timings.get("tokens_per_sec") or 0.0)
        output_tokens = timings.get("output_tokens")

        gpu_util = gpu_stats.avg_gpu_util_percent
        gpu_mem = gpu_stats.max_gpu_mem_percent
        eff_tps = gpu_tps * (((gpu_util or 0.0)) / 100.0)

        cpu_seconds = max(0.0, response_time_s - llm_latency_s)
        total_cost = usd_cost_from_time(
            gpu_seconds=llm_latency_s,
            cpu_seconds=cpu_seconds,
            gpu_cost_per_hour=gpu_cost_per_hour,
            cpu_cost_per_hour=cpu_cost_per_hour,
            fixed_cost_usd=fixed_cost_usd,
        )

        grounding = judge_claims(judge_llm, context=context, answer=answer, max_claims=max_claims)
        rel = judge_relevance(judge_llm, question=q, answer=answer, context=context)

        row = RagEvalRow(
            query=q,
            answer=answer,
            hallucination_rate=float(grounding["hallucination_rate"]),
            groundedness_score=float(grounding["groundedness_score"]),
            answer_relevance_1to5=int(rel["answer_relevance_1to5"] or 0),
            context_relevance_1to5=int(rel["context_relevance_1to5"] or 0),
            confidence=confidence,
            response_time_s=response_time_s,
            llm_latency_s=llm_latency_s,
            gpu_throughput_toks_per_s=gpu_tps,
            eff_gpu_throughput=eff_tps,
            gpu_util_percent=gpu_util,
            gpu_mem_percent=gpu_mem,
            gpu_util_max_percent=gpu_stats.max_gpu_util_percent,
            gpu_mem_avg_percent=gpu_stats.avg_gpu_mem_percent,
            gpu_mem_peak_mb=gpu_stats.max_gpu_mem_mb,
            gpu_mem_torch_peak_mb=gpu_stats.torch_peak_mem_mb,
            gpu_monitor_samples=int(gpu_stats.sample_count),
            total_deployment_cost_usd=total_cost,
            top_source=top_source,
            context_status=str(result.get("context_status", "")),
            answer_mode_used=str(result.get("answer_mode_used", "")),
            used_general_knowledge=bool(result.get("used_general_knowledge", False)),
            retrieved_docs_count=int(result.get("retrieved_docs_count", 0) or 0),
            top_retrieval_score=float(result.get("top_retrieval_score", 0.0) or 0.0),
            avg_retrieval_score=float(result.get("avg_retrieval_score", 0.0) or 0.0),
            query_coverage=float(result.get("query_coverage", 0.0) or 0.0),
        )
        rows.append(row)

        detailed.append({
            "query": q,
            "result": result,
            "timings": timings,
            "output_tokens": output_tokens,
            "grounding": grounding,
            "relevance": rel,
            "gpu": {
                "gpu_util_percent": gpu_util,
                "gpu_mem_percent": gpu_mem,
                "gpu_util_max_percent": gpu_stats.max_gpu_util_percent,
                "gpu_mem_avg_percent": gpu_stats.avg_gpu_mem_percent,
                "gpu_mem_peak_mb": gpu_stats.max_gpu_mem_mb,
                "gpu_mem_torch_peak_mb": gpu_stats.torch_peak_mem_mb,
                "gpu_monitor_samples": int(gpu_stats.sample_count),
            },
            "cost": {
                "gpu_cost_per_hour": gpu_cost_per_hour,
                "cpu_cost_per_hour": cpu_cost_per_hour,
                "fixed_cost_usd": fixed_cost_usd,
                "total_deployment_cost_usd": total_cost,
            },
        })

    return rows, detailed
