import os
import time
from typing import Dict, Any, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)


def _bytes_to_gib(num_bytes: float) -> float:
    return float(num_bytes) / (1024 ** 3)


class LLMManager:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        dtype: str = "auto",
        quantization_mode: str = "4bit",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        do_sample: bool = False,
        trust_remote_code: bool = True,
        no_cpu_offload: bool = True,
        gpu_memory_utilization: float = 0.92,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: str = "auto",
        attn_implementation: str = "auto",
        model_max_length: Optional[int] = None,
    ):
        allowed_quant_modes = {"none", "8bit", "4bit"}
        if quantization_mode not in allowed_quant_modes:
            raise ValueError(
                f"Unsupported quantization_mode={quantization_mode!r}. "
                f"Choose from {sorted(allowed_quant_modes)}."
            )

        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_mode = quantization_mode
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)
        self.trust_remote_code = trust_remote_code
        self.no_cpu_offload = bool(no_cpu_offload)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bool(bnb_4bit_use_double_quant)
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.attn_implementation = attn_implementation
        self.model_max_length = model_max_length

        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.is_chat_model = False
        self.last_memory_estimate = {}

        self.load_model()

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype == "auto":
            if torch.cuda.is_available():
                major = torch.cuda.get_device_properties(0).major
                return torch.bfloat16 if major >= 8 else torch.float16
            return torch.float32

        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return mapping[self.dtype]

    def _resolve_bnb_compute_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        if self.bnb_4bit_compute_dtype == "auto":
            return default_dtype if default_dtype != torch.float32 else torch.float16
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.bnb_4bit_compute_dtype not in mapping:
            raise ValueError(
                "Unsupported bnb_4bit_compute_dtype: "
                f"{self.bnb_4bit_compute_dtype}"
            )
        return mapping[self.bnb_4bit_compute_dtype]

    def _visible_gpu_count(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    def _build_max_memory(self) -> Optional[Dict[Any, str]]:
        if not torch.cuda.is_available():
            return None

        max_memory: Dict[Any, str] = {}
        for gpu_idx in range(torch.cuda.device_count()):
            total_bytes = torch.cuda.get_device_properties(gpu_idx).total_memory
            usable_gib = int((total_bytes / (1024 ** 3)) * self.gpu_memory_utilization)
            usable_gib = max(usable_gib, 1)
            max_memory[gpu_idx] = f"{usable_gib}GiB"

        if self.no_cpu_offload:
            max_memory["cpu"] = "0GiB"

        return max_memory

    def _estimate_parameter_count(self) -> Optional[float]:
        lower = self.model_name.lower()
        candidates = [
            (r"(\d+(?:\.\d+)?)b", 1e9),
            (r"(\d+(?:\.\d+)?)m", 1e6),
        ]
        import re

        for pattern, scale in candidates:
            m = re.search(pattern, lower)
            if m:
                try:
                    return float(m.group(1)) * scale
                except Exception:
                    return None
        return None

    def _estimate_required_gib(self, dtype: torch.dtype) -> Dict[str, Optional[float]]:
        params = self._estimate_parameter_count()
        if params is None:
            return {}

        bytes_per_param_map = {
            "none": 2.0 if dtype in (torch.float16, torch.bfloat16) else 4.0,
            "8bit": 1.0,
            "4bit": 0.5,
        }
        weight_gib = (params * bytes_per_param_map[self.quantization_mode]) / (1024 ** 3)

        overhead_factor_map = {
            "none": 1.10,
            "8bit": 1.15,
            "4bit": 1.20,
        }
        estimated_gib = weight_gib * overhead_factor_map[self.quantization_mode]

        kv_cache_reserve_gib = max(6.0, self.max_new_tokens / 128.0)
        recommended_gib = estimated_gib + kv_cache_reserve_gib
        return {
            "params": params,
            "weight_gib": weight_gib,
            "estimated_runtime_gib": estimated_gib,
            "recommended_total_gib": recommended_gib,
        }

    def _warn_if_tight_fit(self, estimate: Dict[str, Optional[float]]) -> None:
        if not estimate or not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return

        visible_totals = [
            _bytes_to_gib(torch.cuda.get_device_properties(i).total_memory)
            for i in range(torch.cuda.device_count())
        ]
        single_gpu_gib = max(visible_totals)
        needed = estimate.get("recommended_total_gib")
        if needed is None:
            return

        if self.device_map == "auto" and torch.cuda.device_count() > 1:
            total_available = sum(visible_totals)
            print(
                f"[LLMManager] Estimated recommended runtime memory: {needed:.1f} GiB; "
                f"largest visible GPU={single_gpu_gib:.1f} GiB; total visible GPU memory={total_available:.1f} GiB"
            )
        else:
            print(
                f"[LLMManager] Estimated recommended runtime memory: {needed:.1f} GiB; "
                f"visible GPU memory={single_gpu_gib:.1f} GiB"
            )

        if needed > single_gpu_gib and (self.device_map in {"auto", "balanced", "balanced_low_0", "sequential"}):
            print(
                "[LLMManager] WARNING: Estimated memory exceeds a single visible GPU. "
                "Use multiple GPUs, 8-bit, or 4-bit quantization."
            )

    def _assert_no_offload(self) -> None:
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if not hf_device_map:
            return

        bad_locations = []
        for module_name, location in hf_device_map.items():
            if isinstance(location, str) and location.lower() in {"cpu", "disk"}:
                bad_locations.append((module_name, location))

        if bad_locations:
            preview = ", ".join([f"{m}->{loc}" for m, loc in bad_locations[:10]])
            raise RuntimeError(
                "Model was offloaded to CPU/disk, which is not allowed for this run. "
                f"Offloaded modules: {preview}"
            )

    def _build_quant_config(self, torch_dtype: torch.dtype) -> Optional[BitsAndBytesConfig]:
        if self.quantization_mode == "none":
            print("[LLMManager] Quantization disabled")
            return None

        if self.quantization_mode == "8bit":
            print("[LLMManager] Quantization enabled: 8-bit")
            return BitsAndBytesConfig(load_in_8bit=True)

        compute_dtype = self._resolve_bnb_compute_dtype(torch_dtype)
        print(
            "[LLMManager] Quantization enabled: 4-bit "
            f"(quant_type={self.bnb_4bit_quant_type}, double_quant={self.bnb_4bit_use_double_quant}, compute_dtype={compute_dtype})"
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def _resolve_attn_implementation(self) -> Optional[str]:
        if self.attn_implementation == "auto":
            if torch.cuda.is_available():
                return "flash_attention_2"
            return None
        if self.attn_implementation == "none":
            return None
        return self.attn_implementation

    def load_model(self) -> None:
        print(f"\n[LLMManager] Loading model: {self.model_name}")

        if not torch.cuda.is_available():
            print("[LLMManager] WARNING: CUDA is not available. Model will run on CPU.")

        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"[LLMManager] CUDA_VISIBLE_DEVICES={visible_env!r}")
        print(f"[LLMManager] torch.cuda.device_count()={self._visible_gpu_count()}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )

        if self.model_max_length is not None:
            self.tokenizer.model_max_length = int(self.model_max_length)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch_dtype = self._resolve_dtype()
        print(f"[LLMManager] Resolved dtype: {torch_dtype}")

        self.last_memory_estimate = self._estimate_required_gib(torch_dtype)
        if self.last_memory_estimate:
            print(f"[LLMManager] Memory estimate: {self.last_memory_estimate}")
            self._warn_if_tight_fit(self.last_memory_estimate)

        quant_config = self._build_quant_config(torch_dtype)

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.trust_remote_code,
        }

        attn_impl = self._resolve_attn_implementation()
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
            print(f"[LLMManager] attn_implementation={attn_impl}")

        if torch.cuda.is_available():
            model_kwargs["device_map"] = self.device_map
            max_memory = self._build_max_memory()
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory
                print(f"[LLMManager] max_memory={max_memory}")
        else:
            model_kwargs["device_map"] = None

        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        hf_device_map = getattr(self.model, "hf_device_map", None)
        print(f"[LLMManager] hf_device_map={hf_device_map}")

        if self.no_cpu_offload:
            self._assert_no_offload()

        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.is_chat_model = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )

        print(
            "[LLMManager] Detected CHAT model"
            if self.is_chat_model
            else "[LLMManager] Detected COMPLETION model"
        )
        print("[LLMManager] Model ready!\n")

    def _format_prompt(self, prompt: str) -> str:
        if self.is_chat_model:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise long-form reasoning assistant. "
                        "Use retrieved evidence carefully, synthesize across passages, and avoid fabricated claims."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def invoke_timed(self, prompt: str) -> Dict[str, Any]:
        formatted_prompt = self._format_prompt(prompt)

        try:
            prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
        except Exception:
            prompt_tokens = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        raw = self.pipe(formatted_prompt)
        out = raw[0]["generated_text"].strip()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        llm_latency_s = t1 - t0

        try:
            output_tokens = len(self.tokenizer.encode(out))
        except Exception:
            output_tokens = 0

        tokens_per_sec = (output_tokens / llm_latency_s) if llm_latency_s > 0 else 0.0

        return {
            "text": out,
            "llm_latency_s": float(llm_latency_s),
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "tokens_per_sec": float(tokens_per_sec),
            "memory_estimate": self.last_memory_estimate,
        }

    def invoke(self, prompt: str) -> str:
        return self.invoke_timed(prompt)["text"]
