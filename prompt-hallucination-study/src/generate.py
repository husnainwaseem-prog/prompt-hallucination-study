"""Model loading and greedy generation with SmolLM2-1.7B-Instruct.

Greedy decoding (temperature 0) with a fixed seed makes the eval fully
deterministic: re-running the same prompt gives the same answer.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "/home/hatch/workspace/finetune-small-lm/models/base"


class Generator:
    def __init__(self, model_path: str = DEFAULT_MODEL, threads: int | None = None):
        torch.manual_seed(42)
        torch.set_num_threads(threads or max(1, __import__("os").cpu_count() or 2))
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        self.model.config.use_cache = True
        self.model.eval()

    def _messages(self, user_text: str) -> str:
        """Render the user message through the model's chat template."""
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return user_text

    @torch.no_grad()
    def generate(self, user_text: str, max_new_tokens: int = 32) -> str:
        prompt = self._messages(user_text)
        inputs = self.tok(prompt, return_tensors="pt")
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip()


class FakeLLM:
    """Deterministic stand-in used by the tests (never downloads a model).

    Answers a question correctly when the question text contains a known
    keyword, abstains on trap keywords, and otherwise returns a confident
    wrong answer — so scoring behaviour can be tested end to end.
    """

    def __init__(self, answers: dict | None = None):
        self.answers = answers or {}

    def generate(self, user_text: str, max_new_tokens: int = 32) -> str:
        low = user_text.lower()
        if "mars" in low and "first human" in low:
            return "I don't know."
        for key, ans in self.answers.items():
            if key.lower() in low:
                return ans
        return "London"
