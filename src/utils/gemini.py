"""Gemini 3.1 Flash-Lite on the Google API."""

import os
import time
import urllib.error
from pathlib import Path

from dotenv import load_dotenv

from src.utils.base import Detector, image_b64, output_model, post_json, prompt

URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key="


class Gemini(Detector):
    name = "gemini-3.1-flash-lite_google"
    pause = 6

    def __init__(self):
        load_dotenv()
        self.key = os.environ["GEMINI_API_KEY"]

    def predict(self, path: Path, vocab: dict[str, list[str]]) -> dict[str, str]:
        output = output_model(vocab)
        schema = {
            "type": "object",
            "properties": {key: {"type": "string", "enum": vocab[key]} for key in vocab},
            "required": list(vocab),
        }
        body = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt(vocab)},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64(path)}},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 256,
                "thinkingConfig": {"thinkingBudget": 0},
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        reply = _generate(URL + self.key, body)
        content = reply["candidates"][0]["content"]["parts"][0]["text"]
        return output.model_validate_json(content).model_dump(mode="json")


def _generate(url: str, body: dict) -> dict:
    for attempt in range(8):
        try:
            return post_json(url, body, timeout=45)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
            retry = _busy(err) and attempt < 7
            if not retry:
                raise
            time.sleep(35 if getattr(err, "code", None) == 429 else 2 * (attempt + 1))
    raise RuntimeError("Gemini request failed")


def _busy(err: Exception) -> bool:
    if isinstance(err, TimeoutError):
        return True
    if isinstance(err, urllib.error.HTTPError):
        return err.code in (429, 503)
    return isinstance(err, urllib.error.URLError)
