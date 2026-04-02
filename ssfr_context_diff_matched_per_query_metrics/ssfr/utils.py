import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".json", ".csv", ".tsv", ".py",
    ".yaml", ".yml", ".log", ".pdf"
}


def is_text_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in TEXT_EXTENSIONS


def safe_read_text(path: str, encoding: str = "utf-8") -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except Exception:
                from PyPDF2 import PdfReader
            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n\n".join(pages)

        with open(path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if max_chars is None:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "answer", "output", "content", "generated_text", "response"):
            if k in x and isinstance(x[k], str):
                return x[k]
        return str(x)
    if isinstance(x, list):
        if not x:
            return ""
        return _to_text(x[0])
    if hasattr(x, "content") and isinstance(getattr(x, "content"), str):
        return x.content
    return str(x)


def estimate_tokens(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


def call_llm(llm, prompt: str) -> Dict[str, Any]:
    if hasattr(llm, "invoke_timed"):
        out = llm.invoke_timed(prompt)
        if isinstance(out, dict):
            text = _to_text(out.get("text", "")).strip()
            llm_latency_s = float(out.get("llm_latency_s", 0.0))
            prompt_tokens = int(out.get("prompt_tokens", estimate_tokens(prompt)))
            output_tokens = int(out.get("output_tokens", estimate_tokens(text)))
            tokens_per_sec = float(
                out.get(
                    "tokens_per_sec",
                    (output_tokens / llm_latency_s) if llm_latency_s > 0 else 0.0,
                )
            )
            return {
                "text": text,
                "llm_latency_s": llm_latency_s,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "tokens_per_sec": tokens_per_sec,
            }

    t0 = time.perf_counter()
    if hasattr(llm, "invoke"):
        raw = llm.invoke(prompt)
    elif hasattr(llm, "generate"):
        raw = llm.generate(prompt)
    elif callable(llm):
        raw = llm(prompt)
    else:
        raise TypeError("llm must provide .invoke(prompt), .invoke_timed(prompt), .generate(prompt), or be callable.")

    text = _to_text(raw).strip()
    t1 = time.perf_counter()

    prompt_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(text)
    llm_latency_s = float(t1 - t0)
    tokens_per_sec = (output_tokens / llm_latency_s) if llm_latency_s > 0 else 0.0

    return {
        "text": text,
        "llm_latency_s": llm_latency_s,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
    }


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def contains_refusal(ans: str) -> bool:
    a = (ans or "").lower()
    refusal_patterns = [
        "i don't know",
        "i do not know",
        "not enough information",
        "cannot be determined",
        "can not be determined",
        "insufficient information",
        "no relevant context",
        "not provided in the context",
        "not in the provided snippets",
        "not in the snippets",
    ]
    return any(p in a for p in refusal_patterns)


def safe_json_loads(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    if isinstance(s, list):
        if not s:
            return {}
        s = s[0]
    if not isinstance(s, str):
        s = _to_text(s)

    s = (s or "").strip()
    if not s:
        return {}

    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    candidate = s.replace("\n", " ").strip()
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def now_ts() -> float:
    return time.time()


def get_gpu_metrics() -> Dict[str, float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        utils = []
        mems = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utils.append(float(util.gpu))
            mems.append(100.0 * float(mem.used) / float(mem.total))
        pynvml.nvmlShutdown()
        return {
            "gpu_util_percent": sum(utils) / len(utils) if utils else 0.0,
            "gpu_mem_percent": sum(mems) / len(mems) if mems else 0.0,
        }
    except Exception:
        return {"gpu_util_percent": 0.0, "gpu_mem_percent": 0.0}


@dataclass
class QueryGpuStats:
    avg_gpu_util_percent: float = 0.0
    max_gpu_util_percent: float = 0.0
    avg_gpu_mem_percent: float = 0.0
    max_gpu_mem_percent: float = 0.0
    avg_gpu_mem_mb: float = 0.0
    max_gpu_mem_mb: float = 0.0
    torch_peak_mem_mb: float = 0.0
    sample_count: int = 0


class QueryGpuMonitor:
    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.05):
        self.gpu_index = int(gpu_index)
        self.poll_interval_s = float(poll_interval_s)
        self._stop = threading.Event()
        self._thread = None
        self._utils: List[float] = []
        self._mem_percents: List[float] = []
        self._mem_mbs: List[float] = []
        self._started_nvml = False
        self._handle = None
        self._torch_peak_supported = False

    def start(self):
        try:
            import torch
            if torch.cuda.is_available() and self.gpu_index < torch.cuda.device_count():
                with torch.cuda.device(self.gpu_index):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(self.gpu_index)
                    self._torch_peak_supported = True
        except Exception:
            self._torch_peak_supported = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._started_nvml = True
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self._started_nvml = False
        return self

    def _run(self):
        try:
            import pynvml
            while not self._stop.is_set():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    self._utils.append(float(util.gpu))
                    self._mem_percents.append(100.0 * float(mem.used) / float(mem.total))
                    self._mem_mbs.append(float(mem.used) / (1024 ** 2))
                except Exception:
                    pass
                time.sleep(self.poll_interval_s)
        except Exception:
            return

    def stop(self) -> QueryGpuStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, 2 * self.poll_interval_s))

        torch_peak_mb = 0.0
        try:
            import torch
            if self._torch_peak_supported and torch.cuda.is_available() and self.gpu_index < torch.cuda.device_count():
                with torch.cuda.device(self.gpu_index):
                    torch.cuda.synchronize()
                    torch_peak_mb = float(torch.cuda.max_memory_allocated(self.gpu_index) / (1024 ** 2))
        except Exception:
            torch_peak_mb = 0.0

        if self._started_nvml:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

        def _avg(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0

        def _max(xs):
            return float(max(xs)) if xs else 0.0

        return QueryGpuStats(
            avg_gpu_util_percent=_avg(self._utils),
            max_gpu_util_percent=_max(self._utils),
            avg_gpu_mem_percent=_avg(self._mem_percents),
            max_gpu_mem_percent=_max(self._mem_percents),
            avg_gpu_mem_mb=_avg(self._mem_mbs),
            max_gpu_mem_mb=_max(self._mem_mbs),
            torch_peak_mem_mb=torch_peak_mb,
            sample_count=len(self._utils),
        )
