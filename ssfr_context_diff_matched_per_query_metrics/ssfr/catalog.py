import os
import re
import pandas as pd
from typing import Dict, List, Optional, Set
from .utils import is_text_file, safe_read_text, normalize_ws


def summarize_file_content(text: str, max_chars: int = 300) -> str:
    text = normalize_ws(text)
    if not text:
        return "Empty or unreadable file."
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def file_keywords(filename: str) -> str:
    base = os.path.basename(filename).lower()
    stem = os.path.splitext(base)[0]
    tokens = re.split(r"[_\-\s\.]+", stem)
    return " ".join(t for t in tokens if t)


def build_catalog(
    data_dir: str,
    output_path: Optional[str] = None,
    max_summary_chars: int = 300,
    exclude_paths: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
) -> List[Dict]:
    catalog = []
    exclude_abs: Set[str] = {os.path.abspath(p) for p in (exclude_paths or []) if p}
    exclude_base: Set[str] = {os.path.basename(p) for p in (exclude_paths or []) if p}
    exclude_base.update({n for n in (exclude_names or []) if n})

    for root, _, files in os.walk(data_dir):
        for name in sorted(files):
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, data_dir)

            if os.path.abspath(full_path) in exclude_abs or name in exclude_base or rel_path in exclude_base:
                continue

            if not is_text_file(full_path):
                continue

            content = safe_read_text(full_path)
            summary = summarize_file_content(content, max_chars=max_summary_chars)

            catalog.append({
                "file": rel_path,
                "path": full_path,
                "summary": summary,
                "keywords": file_keywords(rel_path),
                "size_bytes": os.path.getsize(full_path) if os.path.exists(full_path) else 0,
            })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(catalog).to_csv(output_path, index=False)

    return catalog