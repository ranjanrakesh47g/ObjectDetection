"""Local Qwen3-VL checkpoints. One process loads one checkpoint."""

import gc
import json
from pathlib import Path

from src.utils.base import MODELS, Detector, blank, output_model, pick, prompt


class Qwen(Detector):
    def __init__(self, repo: str, name: str, *, bits: int | None = None, tools: bool = False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = name
        self.tools = tools
        self.torch = torch
        kwargs = {"cache_dir": str(MODELS), "device_map": "auto"}
        if bits == 4:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            kwargs["dtype"] = torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
        self.processor = AutoProcessor.from_pretrained(repo, cache_dir=str(MODELS), max_pixels=401408)
        self.processor.tokenizer.padding_side = "left"

    def predict(self, path: Path, vocab: dict[str, list[str]]) -> dict[str, str]:
        self._sync()
        pred = self._tools(path, vocab) if self.tools else self._json(path, vocab)
        self._sync()
        return pred

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _sync(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _generate(self, path: Path, text: str, stop: list[str], limit: int, tools: list | None = None) -> str:
        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": str(path.resolve())},
                {"type": "text", "text": text},
            ],
        }]]
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
        }
        if tools:
            kwargs["tools"] = tools
        inputs = self.processor.apply_chat_template(messages, **kwargs).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=False,
                stop_strings=stop,
                tokenizer=self.processor.tokenizer,
            )
        trimmed = [out[len(src):] for src, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _json(self, path: Path, vocab: dict[str, list[str]]) -> dict[str, str]:
        raw = self._generate(path, prompt(vocab), ["}"], 160)
        if raw.count('"') % 2:
            raw += '"'
        if not raw.rstrip().endswith("}"):
            raw += "}"
        start, end = raw.find("{"), raw.rfind("}")
        try:
            obj = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            obj = {}
        return {key: pick(obj.get(key, raw), vocab[key]) for key in vocab}

    def _tools(self, path: Path, vocab: dict[str, list[str]]) -> dict[str, str]:
        output = output_model(vocab)
        tools = [{
            "type": "function",
            "function": {
                "name": "label_image",
                "description": "Choose one foreground, one background, and one relation.",
                "parameters": output.model_json_schema(),
            },
        }]
        raw = self._generate(path, prompt(vocab), ["</tool_call>"], 180, tools)
        blob = raw[raw.find("{"):raw.rfind("}") + 1]
        try:
            obj = json.loads(blob).get("arguments", {})
            if isinstance(obj, str):
                obj = json.loads(obj)
            return output.model_validate(obj).model_dump(mode="json")
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            return blank()
