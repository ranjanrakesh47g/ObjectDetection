"""Qwen vision models hosted on OpenRouter."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.utils.base import Detector, image_b64, output_model, post_json, prompt


class OpenRouter(Detector):
    def __init__(self, model_id: str, name: str):
        load_dotenv()
        self.model_id = model_id
        self.name = name
        self.key = os.environ["OPENROUTER_API_KEY"]

    def predict(self, path: Path, vocab: dict[str, list[str]]) -> dict[str, str]:
        output = output_model(vocab)
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt(vocab)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64(path)}"}},
            ]}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "label", "strict": True, "schema": output.model_json_schema()},
            },
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_completion_tokens": 256,
        }
        reply = post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            body,
            headers={"Authorization": f"Bearer {self.key}"},
            timeout=120,
        )
        content = reply["choices"][0]["message"]["content"]
        return output.model_validate_json(content).model_dump(mode="json")
