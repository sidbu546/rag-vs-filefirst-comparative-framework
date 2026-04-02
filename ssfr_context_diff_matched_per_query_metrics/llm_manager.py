import importlib.util
import time
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LLMManager:
    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        quantization_mode: str = "4bit",
        device_map: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = False,
        trust_remote_code: bool = True,
        attn_implementation: str = "auto",
        model_max_length: int = 4096,
        allow_cpu_offload: bool = False,
        gpu_memory_utilization: float = 0.95,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_use_double_quant: bool = True,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.quantization_mode = quantization_mode
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.model_max_length = model_max_length
        self.allow_cpu_offload = allow_cpu_offload
        self.gpu_memory_utilization = gpu_memory_utilization
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        self.tokenizer = None
        self.model = None
        self.is_chat_model = False

        self.load_model()

    def _resolve_dtype(self, dtype_name: Optional[str] = None):
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": None,
        }
        return mapping[dtype_name or self.dtype]

    def _resolve_attn_implementation(self) -> Optional[str]:
        attn = (self.attn_implementation or "auto").lower()
        if attn in {"none", "auto", ""}:
            return None
        if attn == "flash_attention_2":
            if importlib.util.find_spec("flash_attn") is None:
                print("flash_attn not installed; falling back from flash_attention_2 to sdpa.")
                return "sdpa"
        return attn

    def _build_quant_config(self):
        mode = (self.quantization_mode or "none").lower()
        if mode in {"none", "fp16", "bf16", "full", "unquantized"}:
            return None

        if mode not in {"4bit", "8bit"}:
            raise ValueError(f"Unsupported quantization_mode={self.quantization_mode}")

        compute_dtype = self._resolve_dtype(self.bnb_4bit_compute_dtype)
        if compute_dtype is None:
            compute_dtype = self._resolve_dtype()
        if compute_dtype is None:
            compute_dtype = torch.bfloat16

        return BitsAndBytesConfig(
            load_in_4bit=(mode == "4bit"),
            load_in_8bit=(mode == "8bit"),
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def _estimate_required_gb(self) -> Optional[float]:
        name = self.model_name.lower()
        params_b = None
        if "70b" in name:
            params_b = 70
        elif "72b" in name:
            params_b = 72
        elif "32b" in name:
            params_b = 32
        if params_b is None:
            return None

        mode = (self.quantization_mode or "none").lower()
        if mode == "4bit":
            bytes_per_param = 0.55
        elif mode == "8bit":
            bytes_per_param = 1.05
        else:
            dtype = self._resolve_dtype()
            bytes_per_param = 2.0 if dtype in {torch.float16, torch.bfloat16} else 4.0

        weight_gb = params_b * bytes_per_param
        runtime_multiplier = 1.12 if mode in {"4bit", "8bit"} else 1.18
        return weight_gb * runtime_multiplier

    def _warn_memory_fit(self):
        if not torch.cuda.is_available():
            return
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return

        est = self._estimate_required_gb()
        if est is None:
            return

        usable = total_gb * float(self.gpu_memory_utilization)
        print(f"Estimated model+runtime footprint: ~{est:.1f} GB; usable GPU memory target: ~{usable:.1f} GB")
        if est > usable:
            print("WARNING: selected configuration may not fit cleanly on one GPU. 4bit/8bit are recommended for 70B/72B on a single H200.")

    def load_model(self):
        print(f"\nLoading model: {self.model_name}\n")
        self._warn_memory_fit()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=False,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = self._build_quant_config()
        attn_impl = self._resolve_attn_implementation()

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.trust_remote_code,
        }

        if torch.cuda.is_available():
            # Force full GPU placement for quantized models on a single visible GPU.
            # This avoids Accelerate trying to dispatch some modules to CPU/disk.
            if self.quantization_mode in {"4bit", "8bit"} and torch.cuda.device_count() == 1 and not self.allow_cpu_offload:
                model_kwargs["device_map"] = {"": 0}
                print("[LLMManager] Forcing full GPU placement for quantized model on single GPU")
            else:
                model_kwargs["device_map"] = self.device_map

            if self.allow_cpu_offload:
                model_kwargs["max_memory"] = {
                    0: f"{int(self.gpu_memory_utilization * 100)}%",
                    "cpu": "256GiB",
                }

        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            resolved_dtype = self._resolve_dtype()
            if resolved_dtype is not None:
                model_kwargs["torch_dtype"] = resolved_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self.model.eval()

        self.is_chat_model = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )

        try:
            first_param = next(self.model.parameters())
            print(f"[LLMManager] first param device={first_param.device}")
            print(f"[LLMManager] first param dtype={first_param.dtype}")
        except Exception as e:
            print(f"[LLMManager] could not inspect first param: {e}")

        print(
            f"[LLMManager] quant flags: "
            f"is_loaded_in_8bit={getattr(self.model, 'is_loaded_in_8bit', False)}, "
            f"is_loaded_in_4bit={getattr(self.model, 'is_loaded_in_4bit', False)}"
        )

        print("Detected CHAT model" if self.is_chat_model else "Detected COMPLETION model")
        print(f"Attention backend: {attn_impl or 'default'}")
        print(f"Quantization mode: {self.quantization_mode}")
        print("Model ready!\n")

    @torch.inference_mode()
    def invoke(self, prompt: str) -> str:
        if self.is_chat_model:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a careful long-form question answering assistant. "
                        "Use provided file evidence first, but still answer the question directly even when the evidence is incomplete. "
                        "Do not refuse merely because retrieval is partial. "
                        "Prefer accurate synthesis, cautious wording, and no invented quotations."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = prompt

        max_input_len = self.model_max_length or getattr(self.tokenizer, "model_max_length", 4096)
        if max_input_len is None or max_input_len > 32768:
            max_input_len = 32768

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        )

        if torch.cuda.is_available():
            try:
                first_param_device = next(self.model.parameters()).device
                if str(first_param_device) != "cpu":
                    inputs = {k: v.to(first_param_device) for k, v in inputs.items()}
            except StopIteration:
                pass

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        outputs = self.model.generate(
            **inputs,
            **generate_kwargs,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def invoke_timed(self, prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        text = self.invoke(prompt)
        t1 = time.perf_counter()

        try:
            prompt_tokens = len(self.tokenizer.encode(prompt))
        except Exception:
            prompt_tokens = 0

        try:
            output_tokens = len(self.tokenizer.encode(text))
        except Exception:
            output_tokens = 0

        llm_latency_s = t1 - t0
        tokens_per_sec = output_tokens / llm_latency_s if llm_latency_s > 0 else 0.0

        return {
            "text": text,
            "llm_latency_s": float(llm_latency_s),
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "tokens_per_sec": float(tokens_per_sec),
        }