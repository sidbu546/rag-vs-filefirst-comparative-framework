# Retrieval-Augmented Generation (RAG)

## Context-Aware Question Answering with Retrieval

---

## Overview

This module implements a **Retrieval-Augmented Generation (RAG)** system that answers questions by:

1. Splitting documents into chunks
2. Encoding them into embeddings
3. Retrieving relevant context using vector similarity
4. Generating grounded answers using an LLM

The system is designed for **high-quality, context-grounded QA** while enabling evaluation of accuracy, hallucination, and efficiency.

---

## Key Features

* 🔹 Semantic retrieval using embeddings
* 🔹 Chroma vector database integration
* 🔹 Intelligent chunking with overlap
* 🔹 Balanced retrieval across multiple documents
* 🔹 LLM-based answer generation
* 🔹 Full evaluation pipeline:

  * Hallucination detection
  * Groundedness scoring
  * Relevance scoring
* 🔹 GPU performance + cost tracking

---

## 🏗️ System Pipeline

```
Documents → Chunking → Embeddings → Vector DB
                                       ↓
Query → Retriever → Context → LLM → Answer
                                       ↓
                              Evaluation + Metrics
```

---

## 📂 Components

* **Chunking**

  * Recursive splitting with overlap
  * Preserves semantic continuity

* **Embeddings**

  * Model: `all-MiniLM-L6-v2`
  * Converts text → vector space

* **Vector Store**

  * Chroma DB
  * Cosine similarity search

* **Retriever**

  * Top-K retrieval
  * Balanced across sources
  * Prevents dominance of a single file

* **Generator**

  * LLaMA / Qwen models
  * Context-aware prompting

---

## ⚙️ Running the System

```bash
python run_models.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data_dir ./data \
  --query_file queries.txt \
  --quantization_mode 4bit or 8bit
```

---

## 📊 Evaluation Metrics

### Quality

* Hallucination Rate
* Groundedness Score
* Answer Relevance (1–5)
* Context Relevance (1–5)

---

### Performance

* Response time
* LLM latency
* Tokens/sec

---

### GPU Metrics

* Avg GPU utilization
* Peak GPU utilization
* GPU memory usage
* Torch peak memory

---

### Cost Metrics

* GPU cost
* CPU cost
* Total cost per query

---

## Retrieval Behavior

* Uses semantic similarity for context selection
* Handles multi-document reasoning
* Adapts to:

  * Strong context → grounded answer
  * Weak context → hybrid answer
  * No context → LLM fallback

---

## Notes

* Large models (70B+) require:

  * Higher GPU capacity OR
  * Quantization (4-bit recommended)
* Embeddings can run on CPU to save GPU memory
Model cards used:
meta-llama/Llama-3.3-70B-Instruct
Qwen/Qwen2.5-32B-Instruct
---

## Summary

RAG improves:

* Context grounding
* Factual accuracy
* Multi-document reasoning

but comes with:

* Higher cost
* Additional infrastructure (vector DB)

---
