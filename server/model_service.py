"""Loads and serves the local causal language model."""

import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "SmolLM2-135M-Instruct"
DEFAULT_MODEL_PATH = r"E:\Hosted LLM\models\smollm2-135m"


class ModelService:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path or os.environ.get("LLM_MODEL_PATH", DEFAULT_MODEL_PATH))
        self.device = torch.device("cuda:0")
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self._generation_lock = threading.Lock()

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this server requires the configured NVIDIA GPU")
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        # FastAPI runs sync endpoints in its thread pool; this lock serializes GPU use.
        with self._generation_lock:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            input_tokens = int(inputs["input_ids"].shape[1])

            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            torch.cuda.synchronize(self.device)
            duration = time.perf_counter() - started

            new_tokens = generated[0, input_tokens:]
            output_tokens = int(new_tokens.shape[0])
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            tokens_per_second = output_tokens / duration if duration > 0 else 0.0

            # Drop references to request tensors and generated KV cache promptly.
            del new_tokens, generated, inputs
            return {
                "response": response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "inference_time_seconds": duration,
                "tokens_per_second": tokens_per_second,
            }

    def gpu_stats(self) -> dict[str, Any]:
        props = torch.cuda.get_device_properties(self.device)
        mib = 1024**2
        return {
            "gpu": torch.cuda.get_device_name(self.device),
            "total_vram_mb": round(props.total_memory / mib),
            "allocated_mb": round(torch.cuda.memory_allocated(self.device) / mib),
            "reserved_mb": round(torch.cuda.memory_reserved(self.device) / mib),
        }
