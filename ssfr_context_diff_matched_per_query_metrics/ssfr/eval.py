import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import call_llm, safe_json_loads, split_sentences


GROUNDING_PROMPT = """You are evaluating a file-first QA system.

Task:
Decide whether the CLAIM is supported by the CONTEXT.

Rules:
- Use only the CONTEXT.
- "supported" = the claim is clearly stated or directly implied by the context.
- "partially_supported" = the claim is partly supported but missing some detail, or wording is looser than the context.
- "unsupported" = the claim is not supported by the context.
- Return ONLY valid JSON.
- Do not include markdown fences.

JSON schema:
{{
  "label": "supported|partially_supported|unsupported",
  "evidence": "short quote or phrase from context, or empty string if unsupported"
}}

Context:
{context}

Claim:
{claim}
"""

QA_RELEVANCE_PROMPT = """You are evaluating answer relevance for a file-first QA system.

Task:
Rate how well the ANSWER addresses the QUESTION.

Rules:
- Score 5 = directly and completely answers the question
- Score 4 = mostly answers the question
- Score 3 = partially answers the question
- Score 2 = weakly related or incomplete
- Score 1 = irrelevant or not an answer
- Return ONLY valid JSON.
- Do not include markdown fences.

JSON schema:
{{
  "score": <integer 1-5>,
  "reason": "<short reason>"
}}

Question:
{question}

Answer:
{answer}
"""

CONTEXT_RELEVANCE_PROMPT = """You are evaluating context relevance for a file-first QA system.

Task:
Rate how useful the CONTEXT is for answering the QUESTION.

Rules:
- Score 5 = highly relevant and sufficient
- Score 4 = relevant and mostly sufficient
- Score 3 = somewhat relevant
- Score 2 = weakly relevant
- Score 1 = irrelevant
- Return ONLY valid JSON.
- Do not include markdown fences.

JSON schema:
{{
  "score": <integer 1-5>,
  "reason": "<short reason>"
}}

Question:
{question}

Context:
{context}
"""

LABEL_TO_SCORE = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}


@dataclass
class SSFREvalRow:
    query: str
    answer: str
    context_status: str
    answer_mode_used: str
    used_general_knowledge: bool
    retrieved_docs_count: int
    top_retrieval_score: float
    avg_retrieval_score: float
    query_coverage: float
    hallucination_rate: float
    groundedness_score: float
    answer_relevance_1to5: int
    context_relevance_1to5: int
    confidence: float
    response_time_s: float
    llm_latency_s: float
    gpu_throughput_toks_per_s: float
    eff_gpu_throughput: float
    gpu_util_percent: float
    gpu_mem_percent: float
    gpu_util_max_percent: float
    gpu_mem_avg_percent: float
    gpu_mem_peak_mb: float
    gpu_mem_torch_peak_mb: float
    gpu_monitor_samples: int
    total_deployment_cost_usd: float
    top_source: str


def _judge_json(llm_judge, prompt: str) -> Dict[str, Any]:
    raw = call_llm(llm_judge, prompt)
    obj = safe_json_loads(raw.get("text", raw) if isinstance(raw, dict) else raw)
    return obj if isinstance(obj, dict) else {}


def _clamp_score_1_to_5(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        return 0
    return max(1, min(5, v))


def judge_claims(llm_judge, context: str, answer: str, context_status: str, max_claims: int = 12) -> Dict[str, Any]:
    answer = (answer or "").strip()
    if not answer:
        return {"claims": [], "hallucination_rate": 0.0, "groundedness_score": 0.0, "unsupported_count": 0, "claim_count": 0}
    if context_status == "no_context":
        return {"claims": [], "hallucination_rate": 0.0, "groundedness_score": 0.0, "unsupported_count": 0, "claim_count": 0}

    claims = split_sentences(answer)[:max_claims]
    judged = []
    for c in claims:
        obj = _judge_json(llm_judge, GROUNDING_PROMPT.format(context=context, claim=c))
        label = (obj.get("label") or "").strip().lower()
        if label not in LABEL_TO_SCORE:
            label = "unsupported"
        evidence = (obj.get("evidence") or "").strip()
        judged.append({"claim": c, "label": label, "evidence": evidence})

    if not judged:
        return {"claims": [], "hallucination_rate": 0.0, "groundedness_score": 0.0, "unsupported_count": 0, "claim_count": 0}

    scores = [LABEL_TO_SCORE[j["label"]] for j in judged]
    unsupported = sum(1 for j in judged if j["label"] == "unsupported")
    claim_count = len(judged)
    return {
        "claims": judged,
        "hallucination_rate": float(unsupported / claim_count),
        "groundedness_score": float(np.mean(scores)),
        "unsupported_count": int(unsupported),
        "claim_count": int(claim_count),
    }


def judge_relevance(llm_judge, question: str, answer: str, context: str) -> Dict[str, Any]:
    qa_obj = _judge_json(llm_judge, QA_RELEVANCE_PROMPT.format(question=question, answer=answer))
    ctx_obj = _judge_json(llm_judge, CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context))
    return {
        "answer_relevance_1to5": _clamp_score_1_to_5(qa_obj.get("score", 0)) if qa_obj else 0,
        "answer_relevance_reason": qa_obj.get("reason", "") if qa_obj else "",
        "context_relevance_1to5": _clamp_score_1_to_5(ctx_obj.get("score", 0)) if ctx_obj else 0,
        "context_relevance_reason": ctx_obj.get("reason", "") if ctx_obj else "",
    }


def evaluate_ssfr(queries: List[Dict[str, Any]], runner, judge_llm, max_claims: int = 12):
    rows = []
    detailed = []
    for item in queries:
        q = item["query"]
        result = runner.answer(q)
        answer = result.get("answer", "")
        context = result.get("context", "")
        context_status = result.get("context_status", "no_context")

        grounding = judge_claims(judge_llm, context=context, answer=answer, context_status=context_status, max_claims=max_claims)
        relevance = judge_relevance(judge_llm, question=q, answer=answer, context=context)

        rows.append(SSFREvalRow(
            query=q,
            answer=answer,
            context_status=context_status,
            answer_mode_used=result.get("answer_mode_used", "general_knowledge_only"),
            used_general_knowledge=bool(result.get("used_general_knowledge", True)),
            retrieved_docs_count=int(result.get("retrieved_docs_count", 0)),
            top_retrieval_score=float(result.get("top_retrieval_score", 0.0)),
            avg_retrieval_score=float(result.get("avg_retrieval_score", 0.0)),
            query_coverage=float(result.get("query_coverage", 0.0)),
            hallucination_rate=float(grounding["hallucination_rate"]),
            groundedness_score=float(grounding["groundedness_score"]),
            answer_relevance_1to5=int(relevance["answer_relevance_1to5"] or 0),
            context_relevance_1to5=int(relevance["context_relevance_1to5"] or 0),
            confidence=float(result.get("confidence", 0.0)),
            response_time_s=float(result.get("response_time_s", 0.0)),
            llm_latency_s=float(result.get("llm_latency_s", 0.0)),
            gpu_throughput_toks_per_s=float(result.get("gpu_throughput_toks_per_s", 0.0)),
            eff_gpu_throughput=float(result.get("eff_gpu_throughput", 0.0)),
            gpu_util_percent=float(result.get("gpu_util_percent", 0.0)),
            gpu_mem_percent=float(result.get("gpu_mem_percent", 0.0)),
            gpu_util_max_percent=float(result.get("gpu_util_max_percent", 0.0)),
            gpu_mem_avg_percent=float(result.get("gpu_mem_avg_percent", 0.0)),
            gpu_mem_peak_mb=float(result.get("gpu_mem_peak_mb", 0.0)),
            gpu_mem_torch_peak_mb=float(result.get("gpu_mem_torch_peak_mb", 0.0)),
            gpu_monitor_samples=int(result.get("gpu_monitor_samples", 0)),
            total_deployment_cost_usd=float(result.get("total_deployment_cost_usd", 0.0)),
            top_source=result.get("top_source", ""),
        ))

        detailed.append({
            "query": q,
            "answer": answer,
            "context": context,
            "context_status": context_status,
            "answer_mode_used": result.get("answer_mode_used", "general_knowledge_only"),
            "used_general_knowledge": bool(result.get("used_general_knowledge", True)),
            "read_files": result.get("read_files", []),
            "reasoning_trace": result.get("reasoning_trace", []),
            "sources": result.get("sources", []),
            "grounding": grounding,
            "relevance": relevance,
        })
    return rows, detailed


def summarize_eval(rows: List[SSFREvalRow]) -> Dict[str, Any]:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "avg_hallucination_rate": float(np.mean([r.hallucination_rate for r in rows])),
        "avg_groundedness_score": float(np.mean([r.groundedness_score for r in rows])),
        "avg_answer_relevance_1to5": float(np.mean([r.answer_relevance_1to5 for r in rows])),
        "avg_context_relevance_1to5": float(np.mean([r.context_relevance_1to5 for r in rows])),
        "avg_confidence": float(np.mean([r.confidence for r in rows])),
        "avg_response_time_s": float(np.mean([r.response_time_s for r in rows])),
        "avg_llm_latency_s": float(np.mean([r.llm_latency_s for r in rows])),
        "avg_gpu_throughput_toks_per_s": float(np.mean([r.gpu_throughput_toks_per_s for r in rows])),
        "avg_eff_gpu_throughput": float(np.mean([r.eff_gpu_throughput for r in rows])),
        "avg_gpu_util_percent": float(np.mean([r.gpu_util_percent for r in rows])),
        "avg_gpu_mem_percent": float(np.mean([r.gpu_mem_percent for r in rows])),
        "avg_total_deployment_cost_usd": float(np.mean([r.total_deployment_cost_usd for r in rows])),
    }


def save_results(rows, detailed, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = os.path.join(out_dir, "ssfr_eval.csv")
    detailed_path = os.path.join(out_dir, "ssfr_detailed.txt")
    summary_path = os.path.join(out_dir, "ssfr_summary.csv")
    df.to_csv(csv_path, index=False)
    summary = summarize_eval(rows)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    with open(detailed_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(detailed, start=1):
            f.write("=" * 100 + "\n")
            f.write(f"EXAMPLE {i}\n")
            f.write(f"QUERY: {item['query']}\n\n")
            f.write(f"CONTEXT_STATUS: {item['context_status']}\n")
            f.write(f"ANSWER_MODE_USED: {item['answer_mode_used']}\n")
            f.write(f"USED_GENERAL_KNOWLEDGE: {item['used_general_knowledge']}\n\n")
            f.write(f"ANSWER:\n{item['answer']}\n\n")
            f.write(f"READ_FILES: {item['read_files']}\n\n")
            f.write("REASONING_TRACE:\n")
            for step in item["reasoning_trace"]:
                f.write(f"  step={step.get('step')}\n")
                f.write(f"  mode={step.get('mode', 'unknown')}\n")
                f.write(f"  chosen={step.get('chosen', '')}\n")
                if "raw_selection" in step:
                    f.write(f"  raw_selection={step.get('raw_selection', '')}\n")
                if "candidates" in step:
                    f.write(f"  candidates={step.get('candidates', [])}\n")
                if "top_ranked_candidates" in step:
                    f.write(f"  top_ranked_candidates={step.get('top_ranked_candidates', [])}\n")
                f.write("\n")
            f.write("SOURCES:\n")
            for src in item["sources"]:
                f.write(f"  source={src['source']} score={src['score']}\n")
                f.write(f"  preview={src['preview']}\n")
            f.write("\nGROUNDING:\n")
            for claim in item["grounding"]["claims"]:
                f.write(f"  [{claim['label']}] {claim['claim']}\n")
                f.write(f"  evidence={claim['evidence']}\n")
            f.write("\nRELEVANCE:\n")
            f.write(f"  answer_relevance_1to5={item['relevance']['answer_relevance_1to5']}\n")
            f.write(f"  context_relevance_1to5={item['relevance']['context_relevance_1to5']}\n")
            f.write(f"  answer_relevance_reason={item['relevance'].get('answer_relevance_reason', '')}\n")
            f.write(f"  context_relevance_reason={item['relevance'].get('context_relevance_reason', '')}\n\n")
            f.write("CONTEXT:\n")
            f.write(item["context"][:12000] + "\n\n")
    return csv_path, detailed_path, summary_path, df
