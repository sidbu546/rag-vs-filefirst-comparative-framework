import re
import time
from typing import Dict, Any, List


def now() -> float:
    return time.perf_counter()


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("answer", "output", "text", "content", "generated_text", "response"):
            if k in x and isinstance(x[k], str):
                return x[k]
        return str(x)
    if hasattr(x, "content") and isinstance(getattr(x, "content"), str):
        return x.content
    if isinstance(x, list) and x:
        return _to_text(x[0])
    return str(x)


def _call_llm(llm, prompt: str) -> str:
    if hasattr(llm, "invoke"):
        return _to_text(llm.invoke(prompt))
    if hasattr(llm, "generate"):
        return _to_text(llm.generate(prompt))
    if callable(llm):
        return _to_text(llm(prompt))
    raise TypeError("llm must provide .invoke(prompt), .generate(prompt), or be callable.")


def _call_llm_timed(llm, prompt: str) -> Dict[str, Any]:
    if hasattr(llm, "invoke_timed"):
        out = llm.invoke_timed(prompt)
        out.setdefault("llm_latency_s", None)
        out.setdefault("output_tokens", None)
        out.setdefault("tokens_per_sec", None)
        return out

    t0 = time.perf_counter()
    text = _call_llm(llm, prompt)
    t1 = time.perf_counter()
    return {"text": text, "llm_latency_s": float(t1 - t0), "output_tokens": None, "tokens_per_sec": None}


def _is_complex_prompt(query: str) -> bool:
    q = (query or "").lower()
    signals = [
        "critically examine", "evaluate", "justify", "compare", "analyse", "analyze",
        "discuss", "assess", "role of", "how did", "why did", "to what extent",
        "in 200 words", "in 300 words", "in 350 words", "with examples",
        "were their goals always aligned", "political, social, and structural reasons",
    ]
    question_marks = query.count("?")
    return len(query.split()) >= 12 or any(s in q for s in signals) or question_marks > 1


def _extract_length_hint(query: str) -> str:
    m = re.search(r"\b(\d{2,4})\s+words?\b", query.lower())
    if not m:
        return ""
    try:
        n = int(m.group(1))
    except Exception:
        return ""
    return f"Target length: about {n} words."


def _prepare_context(results: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    sections = []
    used = 0
    for idx, doc in enumerate(results, start=1):
        md = doc.get("metadata") or {}
        source = md.get("source_file", md.get("source", "unknown"))
        page = md.get("page", "unknown")
        chunk_id = md.get("chunk_id", "NA")
        score = doc.get("similarity_score")
        label = f"[Doc {idx} | source={source} | page={page} | chunk_id={chunk_id} | score={score}]"
        content = (doc.get("content") or "").strip()
        block = f"{label}\n{content}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                sections.append(block[:remaining])
            break
        sections.append(block)
        used += len(block)
    return "\n\n".join(sections)


def _estimate_context_strength(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    if not results:
        return {
            "status": "no_context",
            "top_score": 0.0,
            "avg_score": 0.0,
            "retrieved_docs": 0,
            "query_coverage": 0.0,
        }

    query_tokens = {tok for tok in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(tok) >= 4}
    top_scores = [float(r.get("similarity_score") or 0.0) for r in results if r.get("similarity_score") is not None]
    joined = " ".join((r.get("content") or "") for r in results[:4]).lower()
    overlap = sum(1 for tok in query_tokens if tok in joined)
    coverage = (overlap / max(1, len(query_tokens))) if query_tokens else 0.0
    top_score = max(top_scores) if top_scores else 0.0
    avg_score = (sum(top_scores) / len(top_scores)) if top_scores else 0.0

    if top_score >= 0.50 and coverage >= 0.30:
        status = "grounded_context"
    elif top_score >= 0.30 or coverage >= 0.18:
        status = "partial_context"
    else:
        status = "no_context"

    return {
        "status": status,
        "top_score": float(top_score),
        "avg_score": float(avg_score),
        "retrieved_docs": len(results),
        "query_coverage": float(coverage),
    }


def _build_grounded_prompt(query: str, context: str) -> str:
    analytical_guidance = """
If the question is analytical or evaluative:
1. Start with a one-sentence thesis.
2. Organize the answer into clear points or short paragraphs.
3. Use multiple pieces of evidence from the context.
4. Mention both sides when the question asks for evaluation or comparison.
5. End with a concise judgment or synthesis.
6. When you make a factual point from the context, cite it inline as [Doc X].
""" if _is_complex_prompt(query) else "Use concise factual prose and cite supporting statements inline as [Doc X]."

    return f"""You are a careful RAG assistant.

Use the retrieved context as the primary evidence base.
Your answer MUST visibly depend on the context and should therefore differ from a generic textbook answer.
Ground specific claims in the retrieved documents with inline citations like [Doc 1], [Doc 2].
You may use general knowledge only to connect ideas, but do not introduce unsupported factual claims that contradict the context.
Do not say "I don't know" and do not say the context is insufficient unless the evidence is truly missing.
Do not fabricate page numbers or sources.
{_extract_length_hint(query)}
{analytical_guidance}

Retrieved context:
{context}

Question: {query}

Answer:
"""


def _build_partial_prompt(query: str, context: str) -> str:
    return f"""You are a careful assistant answering with mixed evidence.

Use the retrieved context where it is helpful, and use general knowledge to fill reasonable gaps.
Make the final answer direct and complete.
When a point clearly comes from the retrieved context, cite it inline as [Doc X].
When a point is a general background connection not directly shown in the documents, do not cite it.
Do not mention missing context, retrieval quality, or system limitations.
{_extract_length_hint(query)}

Retrieved context:
{context}

Question: {query}

Answer:
"""


def _build_no_context_prompt(query: str) -> str:
    return f"""You are a knowledgeable assistant.

Answer the question using your general knowledge only.
Do not mention documents, context, retrieval, excerpts, or sources.
Do not output citations like [Doc X] because no usable context was retrieved.
For analytical questions, provide a thesis, evidence-based explanation, and conclusion.
{_extract_length_hint(query)}

Question: {query}

Answer:
"""


def rag_advanced(
    query: str,
    retriever,
    llm,
    top_k: int,
    min_score: float = 0.0,
    return_context: bool = False,
    debug: bool = True,
):
    timings: Dict[str, Any] = {}
    t_total0 = now()

    dynamic_top_k = max(top_k, 5) if _is_complex_prompt(query) else top_k
    max_context_chars = 14000 if _is_complex_prompt(query) else 9000

    t0 = now()
    results = retriever.retrieve(query, top_k=dynamic_top_k, score_threshold=min_score)
    timings["retrieve_s"] = now() - t0

    context_state = _estimate_context_strength(results, query)

    if debug:
        print("\n[DEBUG] Query:", query)
        print("[DEBUG] Complex prompt:", _is_complex_prompt(query))
        print("[DEBUG] Retrieved:", len(results), f"(requested_top_k={dynamic_top_k}, score_threshold={min_score})")
        print("[DEBUG] Context state:", context_state)
        for r in results[: min(5, len(results))]:
            md = (r.get("metadata") or {})
            print(
                "  score=", r.get("similarity_score"),
                "source=", md.get("source_file", md.get("source", "unknown")),
                "page=", md.get("page", "NA"),
                "chunk_id=", md.get("chunk_id", "NA"),
                "neighbor=", r.get("is_neighbor", False),
            )

    context = _prepare_context(results, max_chars=max_context_chars) if results else ""

    sources = []
    for doc in results:
        md = doc.get("metadata", {}) or {}
        content = doc.get("content", "") or ""
        sources.append({
            "source": md.get("source_file", md.get("source", "unknown")),
            "page": md.get("page", "unknown"),
            "chunk_id": md.get("chunk_id", None),
            "score": float(doc.get("similarity_score", 0.0) or 0.0),
            "preview": (content[:300] + "...") if len(content) > 300 else content,
            "is_neighbor": bool(doc.get("is_neighbor", False)),
        })

    if context_state["status"] == "grounded_context":
        prompt = _build_grounded_prompt(query, context)
        answer_mode_used = "grounded_with_context"
        used_general_knowledge = False
    elif context_state["status"] == "partial_context":
        prompt = _build_partial_prompt(query, context)
        answer_mode_used = "hybrid_context_plus_knowledge"
        used_general_knowledge = True
    else:
        prompt = _build_no_context_prompt(query)
        answer_mode_used = "general_knowledge_only"
        used_general_knowledge = True

    llm_out = _call_llm_timed(llm, prompt)
    answer = (llm_out.get("text") or "").strip()

    timings["llm_s"] = float(llm_out.get("llm_latency_s") or 0.0)
    timings["output_tokens"] = llm_out.get("output_tokens")
    timings["tokens_per_sec"] = float(llm_out.get("tokens_per_sec") or 0.0)
    timings["total_s"] = now() - t_total0

    confidence = max([float(doc.get("similarity_score", 0.0) or 0.0) for doc in results if doc.get("similarity_score") is not None] + [0.0])

    out = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "timings": timings,
        "context_status": context_state["status"],
        "answer_mode_used": answer_mode_used,
        "used_general_knowledge": used_general_knowledge,
        "retrieved_docs_count": context_state["retrieved_docs"],
        "top_retrieval_score": context_state["top_score"],
        "avg_retrieval_score": context_state["avg_score"],
        "query_coverage": context_state["query_coverage"],
    }
    if return_context:
        out["context"] = context
    return out
