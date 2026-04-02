import argparse
import os

from llm_manager import LLMManager
from queries import QUERIES
from ssfr.catalog import build_catalog
from ssfr.engine import SSFRRunner
from ssfr.eval import evaluate_ssfr, save_results

DEFAULT_DATA_DIR = "/projectnb/cs585/students/siddhank/data_research/data"
DEFAULT_OUTPUT_DIR = "outputs"


def load_queries(query_file: str = ""):
    if not query_file:
        return QUERIES
    items = []
    with open(query_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            q = line.strip()
            if q:
                items.append({"query": q})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--quantization_mode", default="4bit", choices=["none", "4bit", "8bit"])
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb_4bit_compute_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--no_bnb_4bit_double_quant", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa", choices=["auto", "flash_attention_2", "sdpa", "eager", "none"])
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--allow_cpu_offload", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)

    parser.add_argument("--max_file_chars", type=int, default=350000)
    parser.add_argument("--ranking_chars", type=int, default=120000)
    parser.add_argument("--top_k_files", type=int, default=4)
    parser.add_argument("--top_k_snippets_per_file", type=int, default=5)
    parser.add_argument("--snippet_chars", type=int, default=1800)
    parser.add_argument("--snippet_overlap", type=int, default=350)
    parser.add_argument("--lexical_weight", type=float, default=1.0)

    parser.add_argument("--max_new_tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--no_do_sample", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--query_file", default="")
    parser.add_argument("--exclude_query_file_from_corpus", action="store_true", default=True)
    parser.add_argument("--include_query_file_in_corpus", action="store_true")
    parser.add_argument("--max_claims", type=int, default=10)
    args = parser.parse_args()

    quantization_mode = args.quantization_mode
    if args.load_in_8bit:
        quantization_mode = "8bit"
    elif args.load_in_4bit:
        quantization_mode = "4bit"
    elif args.no_4bit and args.quantization_mode == "4bit":
        quantization_mode = "none"

    do_sample = args.do_sample and not args.no_do_sample
    queries = load_queries(args.query_file)

    llm = LLMManager(
        model_name=args.model,
        dtype=args.dtype,
        quantization_mode=quantization_mode,
        device_map="auto",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=do_sample,
        attn_implementation=args.attn_implementation,
        model_max_length=args.model_max_length,
        allow_cpu_offload=args.allow_cpu_offload,
        gpu_memory_utilization=args.gpu_memory_utilization,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=not args.no_bnb_4bit_double_quant,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    exclude_paths = []
    exclude_names = ["catalog.csv", "ssfr_eval.csv", "ssfr_summary.csv", "ssfr_detailed.txt"]
    if args.query_file and not args.include_query_file_in_corpus:
        exclude_paths.append(args.query_file)

    print("\n[1] Building catalog...")
    catalog = build_catalog(
        data_dir=args.data_dir,
        output_path=os.path.join(args.output_dir, "catalog.csv"),
        max_summary_chars=500,
        exclude_paths=exclude_paths,
        exclude_names=exclude_names,
    )
    print(f"Catalog size: {len(catalog)} files")

    print("\n[2] Creating SSFR runner...")
    runner = SSFRRunner(
        llm=llm,
        catalog=catalog,
        max_file_chars=args.max_file_chars,
        ranking_chars=args.ranking_chars,
        top_k_files=args.top_k_files,
        top_k_snippets_per_file=args.top_k_snippets_per_file,
        snippet_chars=args.snippet_chars,
        snippet_overlap=args.snippet_overlap,
        lexical_weight=args.lexical_weight,
        debug=args.debug,
    )

    print("\n[3] Running evaluation...")
    rows, detailed = evaluate_ssfr(
        queries=queries,
        runner=runner,
        judge_llm=llm,
        max_claims=args.max_claims,
    )

    print("\n[4] Saving results...")
    csv_path, detailed_path, summary_path, df = save_results(rows, detailed, out_dir=args.output_dir)

    print("\nSaved:")
    print(csv_path)
    print(detailed_path)
    print(summary_path)

    preview_cols = [
        "query", "context_status", "answer_mode_used", "used_general_knowledge",
        "retrieved_docs_count", "top_retrieval_score", "answer",
        "groundedness_score", "context_relevance_1to5", "confidence", "top_source",
        "gpu_util_percent", "gpu_mem_percent", "gpu_mem_peak_mb", "gpu_mem_torch_peak_mb",
    ]
    print("\nPreview:")
    print(df[preview_cols])


if __name__ == "__main__":
    main()
