import argparse
import json
import os
from dataclasses import asdict

import pandas as pd
import torch

from llm_manager import LLMManager
from rag_eval import evaluate_rag
from rag_retriever_chroma import build_chroma_rag


MODEL_PRESETS = {
    "llama33_70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "dtype": "bfloat16",
        "max_new_tokens": 384,
        "temperature": 0.2,
        "recommended_quant_single_h200": "8bit",
        "notes": "Single H200: use 8-bit or 4-bit. Unquantized usually needs >1 GPU in HF Transformers.",
    },
    "qwen25_72b": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "dtype": "bfloat16",
        "max_new_tokens": 384,
        "temperature": 0.2,
        "recommended_quant_single_h200": "8bit",
        "notes": "Single H200: use 8-bit or 4-bit. Unquantized usually needs >1 GPU in HF Transformers.",
    },
    "qwen25_32b": {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "dtype": "bfloat16",
        "max_new_tokens": 384,
        "temperature": 0.2,
        "recommended_quant_single_h200": "8bit",
        "notes": "Single H200: 4-bit, 8-bit, and often unquantized bf16 are practical for 32B.",
    },
}


def _load_queries(query_file: str):
    path = query_file.lower()
    if path.endswith(".json"):
        with open(query_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [{"query": x if isinstance(x, str) else x["query"]} for x in obj]
        raise ValueError("JSON query file must contain a list.")

    with open(query_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [{"query": q} for q in lines]


def _apply_preset(args):
    if not args.model_preset:
        return args
    preset = MODEL_PRESETS[args.model_preset]
    args.model = preset["model"]
    if args.dtype == "auto":
        args.dtype = preset["dtype"]
    if args.max_new_tokens == parser_defaults["max_new_tokens"]:
        args.max_new_tokens = preset["max_new_tokens"]
    if args.temperature == parser_defaults["temperature"]:
        args.temperature = preset["temperature"]
    return args


def _validate_h200_single_gpu_choice(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name.lower()
    gpu_gib = props.total_memory / (1024 ** 3)
    is_h200_like = "h200" in gpu_name and torch.cuda.device_count() == 1
    is_large_70b_model = any(x in args.model.lower() for x in ["70b", "72b"])

    if is_h200_like and is_large_70b_model and args.quantization_mode == "none":
        print(
            "[run_models] WARNING: single visible H200 detected with an unquantized 70B/72B model. "
            "This is usually too tight for HF Transformers once weights, buffers, and KV cache are included. "
            f"Visible memory={gpu_gib:.1f} GiB. Prefer --quantization_mode 8bit or 4bit."
        )


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="HF model id")
    parser.add_argument("--model_preset", type=str, default="", choices=["", *MODEL_PRESETS.keys()])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)

    parser.add_argument("--quantization_mode", type=str, default="4bit", choices=["none", "8bit", "4bit"])
    parser.add_argument("--load_in_4bit", action="store_true", help="Backward-compatible alias for --quantization_mode 4bit")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=None)
    parser.add_argument("--load_in_8bit", action="store_true", help="Backward-compatible alias for --quantization_mode 8bit")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=None)

    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--no_bnb_4bit_double_quant", action="store_true")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager", "none"])
    parser.add_argument("--model_max_length", type=int, default=None)

    parser.add_argument("--allow_cpu_offload", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)

    parser.add_argument("--data_dir", type=str, default="/projectnb/cs585/students/siddhank/data_research/data")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk_size", type=int, default=700)
    parser.add_argument("--chunk_overlap", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--pool_size", type=int, default=60)
    parser.add_argument("--max_per_source", type=int, default=3)
    parser.add_argument("--preferred_sources", type=str, default="")
    parser.add_argument("--neighbor_window", type=int, default=1)
    parser.add_argument("--lexical_weight", type=float, default=0.20)

    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--no_do_sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=False)

    parser.add_argument("--max_claims", type=int, default=10)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="rag_eval.csv")
    parser.add_argument("--query_file", type=str, default="")

    parser.add_argument("--gpu_cost_per_hour", type=float, default=3.0)
    parser.add_argument("--cpu_cost_per_hour", type=float, default=2.0)
    parser.add_argument("--fixed_cost_usd", type=float, default=2.0)
    return parser


def _resolve_quantization_aliases(args):
    if args.load_in_4bit is True and args.load_in_8bit is True:
        raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit.")
    if args.load_in_4bit is True:
        args.quantization_mode = "4bit"
    elif args.load_in_8bit is True:
        args.quantization_mode = "8bit"
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = _resolve_quantization_aliases(args)
    args = _apply_preset(args)

    print("\n===== ENV =====")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gib = props.total_memory / (1024 ** 3)
        print(f"visible cuda:{i} -> {props.name}, {total_gib:.2f} GiB")
    print("==============\n")

    if args.model_preset:
        print(f"[run_models] Applied preset: {args.model_preset} -> {MODEL_PRESETS[args.model_preset]}")

    _validate_h200_single_gpu_choice(args)

    preferred_sources = [s.strip() for s in (args.preferred_sources or "").split(",") if s.strip()]

    rag_retriever = build_chroma_rag(
        data_dir=args.data_dir,
        exclude_paths=[args.query_file] if args.query_file else None,
        embed_model=args.embed_model,
        collection_name="txt_documents",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        max_per_source=args.max_per_source,
        preferred_sources=preferred_sources,
        neighbor_window=args.neighbor_window,
        lexical_weight=args.lexical_weight,
    )

    llm_manager = LLMManager(
        model_name=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        quantization_mode=args.quantization_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        trust_remote_code=args.trust_remote_code,
        no_cpu_offload=not args.allow_cpu_offload,
        gpu_memory_utilization=args.gpu_memory_utilization,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=not args.no_bnb_4bit_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        attn_implementation=args.attn_implementation,
        model_max_length=args.model_max_length,
    )

    if args.query_file:
        test_queries = _load_queries(args.query_file)
    else:
        test_queries = [
            {"query": "Explain how the idea of a nation evolved in Europe during the 19th century."},
            {"query": "How did Napoleon both promote and suppress nationalism in Europe? Evaluate both sides."},
            {"query": "Why did the 1848 revolutions fail despite widespread support? Analyse political, social, and structural reasons in 300 words."},
        ]

    rows, detailed = evaluate_rag(
        queries=test_queries,
        retriever=rag_retriever,
        rag_llm=llm_manager,
        judge_llm=llm_manager,
        top_k=args.top_k,
        min_score=args.min_score,
        max_claims=args.max_claims,
        gpu_index=args.gpu_index,
        gpu_cost_per_hour=args.gpu_cost_per_hour,
        cpu_cost_per_hour=args.cpu_cost_per_hour,
        fixed_cost_usd=args.fixed_cost_usd,
    )

    df = pd.DataFrame([asdict(r) for r in rows])

    cols = [
        "query",
        "answer",
        "hallucination_rate",
        "groundedness_score",
        "answer_relevance_1to5",
        "context_relevance_1to5",
        "confidence",
        "response_time_s",
        "llm_latency_s",
        "gpu_throughput_toks_per_s",
        "eff_gpu_throughput",
        "gpu_util_percent",
        "gpu_mem_percent",
        "gpu_util_max_percent",
        "gpu_mem_avg_percent",
        "gpu_mem_peak_mb",
        "gpu_mem_torch_peak_mb",
        "gpu_monitor_samples",
        "total_deployment_cost_usd",
        "top_source",
        "context_status",
        "answer_mode_used",
        "used_general_knowledge",
        "retrieved_docs_count",
        "top_retrieval_score",
        "avg_retrieval_score",
        "query_coverage",
    ]

    df_view = df[cols]

    print("\n=== RESULTS ===")
    print(df_view)

    df_view.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV: {args.out_csv}")


parser_defaults = {
    "max_new_tokens": 320,
    "temperature": 0.2,
}


if __name__ == "__main__":
    main()
