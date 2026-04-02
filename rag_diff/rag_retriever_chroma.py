import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
except Exception:
    pass

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def _normalize_path(path: str) -> str:
    return str(Path(path).resolve())


def load_documents(data_dir: str, exclude_paths: Optional[List[str]] = None) -> List[Any]:
    """
    Loads .txt and .pdf files recursively.
    PDF pages are preserved in metadata so answer prompts can cite page-level evidence.
    """
    docs: List[Any] = []
    excluded = {str(Path(x).resolve()) for x in (exclude_paths or [])}

    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        silent_errors=True,
    )
    docs.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False,
        silent_errors=True,
    )
    docs.extend(pdf_loader.load())

    filtered_docs = []
    for d in docs:
        try:
            src_resolved = str(Path((d.metadata or {}).get("source", "")).resolve())
        except Exception:
            src_resolved = (d.metadata or {}).get("source", "")
        if src_resolved in excluded:
            continue
        filtered_docs.append(d)

    docs = filtered_docs

    for d in docs:
        md = d.metadata or {}
        src = md.get("source", "")
        md["source"] = src
        md["source_file"] = os.path.basename(src) if src else "unknown"
        if "page" in md and md["page"] is not None:
            try:
                md["page"] = int(md["page"]) + 1
            except Exception:
                pass
        d.metadata = md

    print(f"[Loader] Loaded {len(docs)} documents from: {data_dir}")
    return docs


def split_documents(
    documents: List[Any],
    chunk_size: int = 700,
    chunk_overlap: int = 180,
) -> List[Any]:
    """
    Uses structure-aware splitting that preserves paragraph flow better for analytical questions.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        md = chunk.metadata or {}
        md["chunk_id"] = idx
        md["char_len"] = len(chunk.page_content or "")
        chunk.metadata = md
    print(f"[Splitter] Created {len(chunks)} chunks")
    return chunks


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )


class VectorStore:
    def __init__(self, collection_name: str = "txt_documents"):
        self.client = chromadb.Client()
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add_documents(self, chunks: List[Any], embeddings: np.ndarray) -> None:
        ids = [str(uuid.uuid4()) for _ in chunks]
        docs = [c.page_content for c in chunks]
        metas = [c.metadata for c in chunks]
        self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings.tolist())


def add_documents_in_batches(
    vectorstore: VectorStore,
    chunks: List[Any],
    embeddings: np.ndarray,
    batch_size: int = 5000,
) -> None:
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embs = embeddings[i:i + batch_size]
        vectorstore.add_documents(batch_chunks, batch_embs)
        print(f"[Chroma] Added batch {i} → {i + len(batch_chunks)}")


class ChromaBalancedRAGRetriever:
    """
    Dense retrieval + lightweight hybrid reranking.

    Improvements aimed at complex prompts:
    - pulls a larger candidate pool
    - balances across sources/pages
    - boosts evidence that overlaps with the analytical question terms
    - can return neighboring chunks to preserve argument continuity
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        pool_size: int = 60,
        max_per_source: int = 3,
        preferred_sources: Optional[List[str]] = None,
        neighbor_window: int = 1,
        lexical_weight: float = 0.20,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.pool_size = int(pool_size)
        self.max_per_source = int(max_per_source)
        self.preferred_sources = preferred_sources or []
        self.neighbor_window = int(max(0, neighbor_window))
        self.lexical_weight = float(max(0.0, lexical_weight))
        print(
            f"[Retriever] pool_size={self.pool_size} max_per_source={self.max_per_source} "
            f"preferred={self.preferred_sources} neighbor_window={self.neighbor_window}"
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        out = []
        for tok in (text or "").lower().replace("\n", " ").split():
            tok = "".join(ch for ch in tok if ch.isalnum())
            if len(tok) >= 3:
                out.append(tok)
        return out

    def _hybrid_score(self, query: str, doc_txt: str, dense_similarity: float) -> float:
        q_tokens = set(self._tokenize(query))
        d_tokens = set(self._tokenize(doc_txt))
        lexical = (len(q_tokens & d_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
        return float((1.0 - self.lexical_weight) * dense_similarity + self.lexical_weight * lexical)

    def _candidate_pool(self, query: str, top_k: int) -> List[Tuple[float, str, str, Dict[str, Any], float]]:
        q_emb = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=max(top_k, self.pool_size),
        )
        if not results.get("documents") or not results["documents"][0]:
            return []

        candidates = []
        for doc_id, doc_txt, md, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            dense_similarity = 1.0 - float(dist)
            hybrid_score = self._hybrid_score(query, doc_txt, dense_similarity)
            candidates.append((hybrid_score, doc_id, doc_txt, md or {}, float(dist)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    def _fetch_neighbor_chunks(self, source_file: str, page: Optional[int], chunk_id: Optional[int]) -> List[Dict[str, Any]]:
        if self.neighbor_window <= 0 or chunk_id is None:
            return []

        neighbors: List[Dict[str, Any]] = []
        try:
            where = {"$and": [{"source_file": source_file}]}
            if page is not None:
                where["$and"].append({"page": page})
            res = self.vector_store.collection.get(where=where, include=["documents", "metadatas"])
            docs = res.get("documents", []) or []
            metas = res.get("metadatas", []) or []
            ids = res.get("ids", []) or []
            pairs = []
            for _id, txt, md in zip(ids, docs, metas):
                cid = (md or {}).get("chunk_id")
                if isinstance(cid, int):
                    pairs.append((_id, txt, md))
            pairs.sort(key=lambda x: x[2].get("chunk_id", 0))
            for _id, txt, md in pairs:
                cid = md.get("chunk_id")
                if cid is None or cid == chunk_id:
                    continue
                if abs(cid - chunk_id) <= self.neighbor_window:
                    neighbors.append(
                        {
                            "id": _id,
                            "content": txt,
                            "metadata": md,
                            "similarity_score": None,
                            "distance": None,
                            "is_neighbor": True,
                        }
                    )
        except Exception:
            return []
        return neighbors

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        candidates = self._candidate_pool(query, top_k=top_k)
        if not candidates:
            return []

        out: List[Dict[str, Any]] = []
        per_src: Dict[str, int] = {}
        seen_ids = set()

        for sim, doc_id, doc_txt, md, dist in candidates:
            if sim < score_threshold:
                continue
            src = md.get("source_file", "unknown")
            if per_src.get(src, 0) >= self.max_per_source:
                continue
            if doc_id in seen_ids:
                continue

            item = {
                "id": doc_id,
                "content": doc_txt,
                "metadata": md,
                "similarity_score": float(sim),
                "distance": float(dist),
                "is_neighbor": False,
            }
            out.append(item)
            seen_ids.add(doc_id)
            per_src[src] = per_src.get(src, 0) + 1
            if len(out) >= top_k:
                break

        if self.preferred_sources and len(out) < top_k:
            picked_sources = {d["metadata"].get("source_file", "unknown") for d in out}
            for pref in self.preferred_sources:
                if len(out) >= top_k:
                    break
                if pref in picked_sources:
                    continue
                for sim, doc_id, doc_txt, md, dist in candidates:
                    if md.get("source_file", "unknown") != pref or sim < score_threshold:
                        continue
                    if doc_id in seen_ids:
                        continue
                    out.append(
                        {
                            "id": doc_id,
                            "content": doc_txt,
                            "metadata": md,
                            "similarity_score": float(sim),
                            "distance": float(dist),
                            "is_neighbor": False,
                        }
                    )
                    seen_ids.add(doc_id)
                    picked_sources.add(pref)
                    break

        expanded: List[Dict[str, Any]] = []
        for item in out:
            expanded.append(item)
            md = item.get("metadata") or {}
            expanded.extend(
                n for n in self._fetch_neighbor_chunks(
                    source_file=md.get("source_file", "unknown"),
                    page=md.get("page"),
                    chunk_id=md.get("chunk_id"),
                )
                if n["id"] not in seen_ids
            )
            for n in expanded:
                seen_ids.add(n["id"])

        return expanded


def build_chroma_rag(
    data_dir: str,
    exclude_paths: Optional[List[str]] = None,
    embed_model: str = "all-MiniLM-L6-v2",
    collection_name: str = "txt_documents",
    chunk_size: int = 700,
    chunk_overlap: int = 180,
    batch_size: int = 5000,
    pool_size: int = 60,
    max_per_source: int = 3,
    preferred_sources: Optional[List[str]] = None,
    neighbor_window: int = 1,
    lexical_weight: float = 0.20,
):
    docs = load_documents(data_dir, exclude_paths=exclude_paths)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    emb_mgr = EmbeddingManager(embed_model)
    texts = [c.page_content for c in chunks]
    embs = emb_mgr.generate_embeddings(texts)

    vs = VectorStore(collection_name=collection_name)
    add_documents_in_batches(vs, chunks, embs, batch_size=batch_size)

    retriever = ChromaBalancedRAGRetriever(
        vector_store=vs,
        embedding_manager=emb_mgr,
        pool_size=pool_size,
        max_per_source=max_per_source,
        preferred_sources=preferred_sources,
        neighbor_window=neighbor_window,
        lexical_weight=lexical_weight,
    )
    return retriever
