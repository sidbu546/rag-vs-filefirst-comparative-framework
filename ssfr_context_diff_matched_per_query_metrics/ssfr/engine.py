import re
from typing import Any, Dict, List, Tuple

from .loader import load_file_content
from .utils import call_llm, estimate_tokens, now_ts, QueryGpuMonitor


GROUNDED_ANSWER_PROMPT = """You are answering a complex question using retrieved file evidence.

Rules:
1. Answer the question directly and clearly.
2. Use the evidence excerpts as your primary support.
3. Include inline citations like [Doc 1], [Doc 2] when using the retrieved evidence.
4. For analytical or history prompts, use a thesis, key supporting points, and a short conclusion.
5. Do not mention retrieval, snippets, or missing context.
6. Do not invent quotations or citations.

Question:
{question}

Evidence excerpts:
{context}

Answer:
"""

PARTIAL_ANSWER_PROMPT = """You are answering a complex question with partial but useful file evidence.

Rules:
1. Answer the question directly.
2. Use the evidence excerpts where they help, with inline citations like [Doc 1] when appropriate.
3. Fill the remaining gaps with strong background knowledge.
4. Make the final answer complete and analytical.
5. Do not say the context is insufficient or mention retrieval.
6. Do not invent quotations.

Question:
{question}

Evidence excerpts:
{context}

Answer:
"""

NO_CONTEXT_ANSWER_PROMPT = """You are answering a complex question without usable retrieved context.

Rules:
1. Answer directly using strong general knowledge.
2. Do not use any document-style citations such as [Doc 1].
3. Do not mention missing context, missing excerpts, or retrieval.
4. For analytical or history prompts, provide a thesis, key supporting points, and a short conclusion.
5. For counterfactual or hypothetical questions, answer as a reasoned inference.

Question:
{question}

Answer:
"""

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "at",
    "for", "and", "or", "but", "with", "by", "from", "as", "about", "into",
    "what", "who", "when", "where", "why", "how", "does", "do", "did",
    "give", "tell", "me", "please", "main", "one", "two", "three"
}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def simplify_query(query: str) -> str:
    toks = tokenize(query)
    toks = [t for t in toks if t not in STOPWORDS]
    return " ".join(toks)


def leading_keyphrase(query: str, max_words: int = 8) -> str:
    toks = [t for t in tokenize(query) if t not in STOPWORDS]
    return " ".join(toks[:max_words])


def exact_phrase_score(query: str, text: str) -> float:
    q = normalize_text(query)
    t = normalize_text(text)
    if not q or not t:
        return 0.0
    return 1000.0 if q in t else 0.0


def token_overlap_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def token_frequency_score(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0
    total = sum(t_tokens.count(tok) for tok in q_tokens)
    return total / max(1, len(q_tokens))


def ordered_token_span_score(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0
    pos = 0
    matched = 0
    for qt in q_tokens:
        found = False
        for i in range(pos, len(t_tokens)):
            if t_tokens[i] == qt:
                matched += 1
                pos = i + 1
                found = True
                break
        if not found:
            break
    return matched / max(1, len(q_tokens))


def split_into_chunks(text: str, chunk_chars: int = 1800, overlap: int = 350) -> List[str]:
    text = text or ""
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class SSFRRunner:
    def __init__(
        self,
        llm,
        catalog: List[Dict],
        max_file_chars: int = 350000,
        ranking_chars: int = 120000,
        top_k_files: int = 4,
        top_k_snippets_per_file: int = 5,
        snippet_chars: int = 1800,
        snippet_overlap: int = 350,
        lexical_weight: float = 1.0,
        debug: bool = False,
    ):
        self.llm = llm
        self.catalog = catalog
        self.file_map = {item["file"]: item for item in catalog}
        self.max_file_chars = max_file_chars
        self.ranking_chars = ranking_chars
        self.top_k_files = top_k_files
        self.top_k_snippets_per_file = top_k_snippets_per_file
        self.snippet_chars = snippet_chars
        self.snippet_overlap = snippet_overlap
        self.lexical_weight = lexical_weight
        self.debug = debug

    def _ranking_text(self, item: Dict) -> str:
        summary = item.get("summary", "")
        full_text = load_file_content(item, max_chars=self.ranking_chars)
        keywords = item.get("keywords", "")
        return f"{item['file']}\n{keywords}\n{summary}\n{full_text}"

    def _looks_like_question_bank(self, item: Dict, ranking_text: str) -> bool:
        file_name = str(item.get("file", "")).lower()
        probe = (file_name + "\n" + ranking_text[:4000]).lower()
        patterns = [
            "question bank", "sample questions", "prompts", "essay questions", "write a 500",
            "explain how", "discuss the", "suppose", "imagine", "analyse", "analyze", "evaluate",
        ]
        numbered = len(re.findall(r"(?:^|\n)\s*\d+[\.)]\s", probe))
        return any(p in probe for p in patterns) or numbered >= 4

    def _score_variant(self, query: str, text: str) -> float:
        if not query.strip() or not text.strip():
            return 0.0
        return (
            exact_phrase_score(query, text)
            + self.lexical_weight * 12.0 * token_overlap_score(query, text)
            + self.lexical_weight * 8.0 * token_frequency_score(query, text)
            + self.lexical_weight * 10.0 * ordered_token_span_score(query, text)
        )

    def _rank_text_against_question(self, question: str, text: str) -> float:
        simplified = simplify_query(question)
        keyphrase = leading_keyphrase(question, max_words=8)
        full_score = self._score_variant(question, text)
        simplified_score = self._score_variant(simplified, text)
        keyphrase_score = self._score_variant(keyphrase, text)
        return max(full_score, simplified_score * 1.15, keyphrase_score * 1.25)

    def _rank_file(self, question: str, item: Dict) -> float:
        ranking_text = self._ranking_text(item)
        score = self._rank_text_against_question(question, ranking_text)
        if self._looks_like_question_bank(item, ranking_text):
            score *= 0.08
        if str(item.get("file", "")).lower().endswith(".pdf"):
            score *= 1.15
        return score

    def _rank_all_files(self, question: str) -> List[Tuple[str, float]]:
        scored = []
        for item in self.catalog:
            fname = item["file"]
            if ".ipynb_checkpoints" in fname:
                continue
            score = self._rank_file(question, item)
            scored.append((fname, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _extract_best_snippets(self, question: str, item: Dict) -> List[Tuple[str, float]]:
        full_text = load_file_content(item, max_chars=self.max_file_chars)
        chunks = split_into_chunks(full_text, chunk_chars=self.snippet_chars, overlap=self.snippet_overlap)
        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            score = self._rank_text_against_question(question, chunk)
            scored_chunks.append((idx, chunk, score))
        scored_chunks.sort(key=lambda x: x[2], reverse=True)
        chosen = []
        seen = set()
        for idx, chunk, score in scored_chunks:
            for neighbor in (idx - 1, idx, idx + 1):
                if 0 <= neighbor < len(chunks) and neighbor not in seen:
                    neighbor_chunk = chunks[neighbor]
                    neighbor_score = self._rank_text_against_question(question, neighbor_chunk)
                    chosen.append((neighbor_chunk, neighbor_score))
                    seen.add(neighbor)
                if len(chosen) >= self.top_k_snippets_per_file:
                    break
            if len(chosen) >= self.top_k_snippets_per_file:
                break
        chosen.sort(key=lambda x: x[1], reverse=True)
        if self.debug:
            print(f"\n[DEBUG] Top snippets from {item['file']}:")
            for idx, (chunk, score) in enumerate(chosen[:5], start=1):
                preview = chunk[:220].replace("\n", " ")
                print(f"  {idx}. score={score:.3f} preview={preview}")
        return chosen[:self.top_k_snippets_per_file]

    def _build_sources(self, question: str, chosen_files: List[str]) -> List[Dict[str, Any]]:
        sources = []
        for fname in chosen_files:
            item = self.file_map[fname]
            preview = load_file_content(item, max_chars=400)
            score = self._rank_file(question, item)
            sources.append({"source": fname, "page": "N/A", "score": score, "preview": preview})
        sources.sort(key=lambda x: x["score"], reverse=True)
        return sources

    def _determine_context_status(self, ranked: List[Tuple[str, float]], snippet_scores: List[float]) -> str:
        if not ranked or not snippet_scores:
            return "no_context"
        top_file_score = ranked[0][1]
        top_snippet = max(snippet_scores) if snippet_scores else 0.0
        mean_snippet = sum(snippet_scores) / max(1, len(snippet_scores))
        if top_file_score >= 12.0 and top_snippet >= 6.5 and mean_snippet >= 3.0:
            return "grounded_context"
        if top_file_score >= 4.0 and top_snippet >= 1.8:
            return "partial_context"
        return "no_context"

    def _build_prompt(self, question: str, context: str, context_status: str) -> Tuple[str, str, bool]:
        if context_status == "grounded_context":
            return GROUNDED_ANSWER_PROMPT.format(question=question, context=context), "grounded_with_context", False
        if context_status == "partial_context":
            return PARTIAL_ANSWER_PROMPT.format(question=question, context=context), "hybrid_context_plus_knowledge", True
        return NO_CONTEXT_ANSWER_PROMPT.format(question=question), "general_knowledge_only", True

    def answer(self, question: str) -> Dict[str, Any]:
        overall_t0 = now_ts()
        monitor = QueryGpuMonitor().start()
        ranked = self._rank_all_files(question)
        if self.debug:
            print(f"\n[DEBUG] Top ranked files for query: {question}")
            for fname, score in ranked[:10]:
                print(f"  {fname}: {score:.3f}")

        chosen_files = [fname for fname, _ in ranked[:self.top_k_files]]
        snippet_blocks = []
        snippet_scores: List[float] = []
        reasoning_trace = [{
            "step": 1,
            "mode": "deterministic_file_first_ranking",
            "chosen_files": chosen_files,
            "top_ranked_candidates": ranked[:10],
            "original_query": question,
            "simplified_query": simplify_query(question),
            "keyphrase_query": leading_keyphrase(question, max_words=8),
        }]

        for fname in chosen_files:
            item = self.file_map[fname]
            best_snippets = self._extract_best_snippets(question, item)
            reasoning_trace.append({
                "step": len(reasoning_trace) + 1,
                "mode": "snippet_extraction_with_neighbors",
                "file": fname,
                "top_snippet_scores": [score for _, score in best_snippets],
            })
            for idx, (snippet, score) in enumerate(best_snippets, start=1):
                snippet_scores.append(score)
                snippet_blocks.append(f"[Doc {len(snippet_blocks) + 1} | Source: {fname} | excerpt {idx} | score={score:.3f}]\n{snippet}")

        context = "\n\n".join(snippet_blocks)
        context_status = self._determine_context_status(ranked, snippet_scores)
        answer_prompt, answer_mode_used, used_general_knowledge = self._build_prompt(question, context, context_status)

        llm_result = call_llm(self.llm, answer_prompt)
        answer = (llm_result.get("text") or "").strip()

        refusal_markers = [
            "i don't know", "i do not know", "provided context", "provided excerpts",
            "provided file excerpts", "insufficient information", "not enough information",
            "not contained in the excerpts", "cannot be determined from the excerpts",
        ]
        lowered = answer.lower()
        if any(m in lowered for m in refusal_markers):
            rescue_prompt = NO_CONTEXT_ANSWER_PROMPT.format(question=question)
            rescue = call_llm(self.llm, rescue_prompt)
            rescue_text = (rescue.get("text") or "").strip()
            if rescue_text:
                answer = rescue_text
                llm_result = rescue
                context_status = "no_context"
                answer_mode_used = "general_knowledge_only"
                used_general_knowledge = True

        overall_t1 = now_ts()
        gpu_stats = monitor.stop()
        response_time_s = overall_t1 - overall_t0
        llm_latency_s = float(llm_result.get("llm_latency_s", 0.0))
        prompt_tokens = int(llm_result.get("prompt_tokens", estimate_tokens(answer_prompt)))
        output_tokens = int(llm_result.get("output_tokens", estimate_tokens(answer)))
        gpu_throughput_toks_per_s = float(llm_result.get("tokens_per_sec", 0.0))
        gpu_util_percent = float(gpu_stats.avg_gpu_util_percent)
        gpu_mem_percent = float(gpu_stats.max_gpu_mem_percent)
        eff_gpu_throughput = gpu_throughput_toks_per_s * (gpu_util_percent / 100.0)

        sources = self._build_sources(question, chosen_files)
        top_score = ranked[0][1] if ranked else 0.0
        avg_retrieval_score = sum(score for _, score in ranked[:self.top_k_files]) / max(1, len(chosen_files)) if chosen_files else 0.0
        query_coverage = min(1.0, max(snippet_scores) / 10.0) if snippet_scores else 0.0
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = max(0.1, min(1.0, (top_score - second_score + 5.0) / 10.0)) if ranked else 0.1
        if context_status == "grounded_context":
            confidence = min(1.0, max(confidence, 0.75))
        elif context_status == "partial_context":
            confidence = min(0.85, max(confidence, 0.45))
        else:
            confidence = min(0.55, max(confidence * 0.7, 0.2))
        top_source = sources[0]["source"] if sources and context_status != "no_context" else ""

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "context": context,
            "context_status": context_status,
            "answer_mode_used": answer_mode_used,
            "used_general_knowledge": bool(used_general_knowledge),
            "retrieved_docs_count": len(chosen_files) if context_status != "no_context" else 0,
            "top_retrieval_score": float(top_score),
            "avg_retrieval_score": float(avg_retrieval_score),
            "query_coverage": float(query_coverage),
            "read_files": chosen_files,
            "reasoning_trace": reasoning_trace,
            "response_time_s": float(response_time_s),
            "llm_latency_s": llm_latency_s,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "gpu_throughput_toks_per_s": gpu_throughput_toks_per_s,
            "eff_gpu_throughput": float(eff_gpu_throughput),
            "gpu_util_percent": gpu_util_percent,
            "gpu_mem_percent": gpu_mem_percent,
            "gpu_util_max_percent": float(gpu_stats.max_gpu_util_percent),
            "gpu_mem_avg_percent": float(gpu_stats.avg_gpu_mem_percent),
            "gpu_mem_peak_mb": float(gpu_stats.max_gpu_mem_mb),
            "gpu_mem_torch_peak_mb": float(gpu_stats.torch_peak_mem_mb),
            "gpu_monitor_samples": int(gpu_stats.sample_count),
            "total_deployment_cost_usd": 2.0,
            "top_source": top_source,
        }
