import re
import json
import numpy as np
from typing import Dict, Any, List

from rag_core import _call_llm

GROUNDING_PROMPT = """You are evaluating a RAG system.

Given:
- Context: passages retrieved from a knowledge base
- Claim: a single sentence from the model answer

Task:
Decide if the claim is FULLY supported by the Context.
- "supported": claim is directly stated or unambiguously implied by context
- "partially_supported": some support but missing key specifics / not fully justified
- "unsupported": not supported or contradicted by context

Return ONLY valid JSON with keys:
{{
  "label": "supported|partially_supported|unsupported",
  "evidence": "short quote or phrase from context (or empty if unsupported)"
}}

Context:
{context}

Claim:
{claim}
"""

QA_RELEVANCE_PROMPT = """You are evaluating a RAG system.

Given:
- Question
- Answer

Score how well the Answer addresses the Question, regardless of grounding.
Return ONLY valid JSON:
{{
  "score": <integer 1-5>,
  "reason": "<short reason>"
}}

Question:
{question}

Answer:
{answer}
"""

CONTEXT_RELEVANCE_PROMPT = """You are evaluating a RAG system.

Given:
- Question
- Context

Score how relevant the Context is to answering the Question.
Return ONLY valid JSON:
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

def safe_json_loads(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]

def judge_claims(llm_judge, context: str, answer: str, max_claims: int = 12) -> Dict[str, Any]:
    claims = split_sentences(answer)[:max_claims]
    judged = []

    for c in claims:
        p = GROUNDING_PROMPT.format(context=context, claim=c)
        raw = _call_llm(llm_judge, p)
        obj = safe_json_loads(raw)

        label = (obj.get("label") or "").strip().lower()
        if label not in LABEL_TO_SCORE:
            label = "unsupported"
            obj["evidence"] = ""

        judged.append({"claim": c, "label": label, "evidence": obj.get("evidence", "")})

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
    qa_raw = _call_llm(llm_judge, QA_RELEVANCE_PROMPT.format(question=question, answer=answer))
    qa_obj = safe_json_loads(qa_raw)
    qa_score = int(qa_obj.get("score", 0) or 0)

    ctx_raw = _call_llm(llm_judge, CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context))
    ctx_obj = safe_json_loads(ctx_raw)
    ctx_score = int(ctx_obj.get("score", 0) or 0)

    return {
        "answer_relevance_1to5": qa_score,
        "answer_relevance_reason": qa_obj.get("reason", ""),
        "context_relevance_1to5": ctx_score,
        "context_relevance_reason": ctx_obj.get("reason", ""),
    }