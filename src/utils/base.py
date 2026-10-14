"""Shared label schema and the detector contract."""

import json
import urllib.request
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
KEYS = ("foreground", "background", "relation")


class Detector:
    """Load one model in ``__init__``. Label one image in ``predict``."""

    name: str
    pause: float = 0

    def predict(self, path: Path, vocab: dict[str, list[str]]) -> dict[str, str]:
        raise NotImplementedError

    def close(self) -> None:
        """Release weights before the next model loads."""


def prompt(vocab: dict[str, list[str]]) -> str:
    lines = ["Label this image. When two labels both fit, choose the more specific one."]
    lines += [f"{key}: {', '.join(vocab[key])}" for key in KEYS]
    return "\n".join(lines)


def _enum(name: str, values: list[str]) -> type[Enum]:
    return Enum(name, {value: value for value in values})


def output_model(vocab: dict[str, list[str]]) -> type[BaseModel]:
    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        relation: _enum("relation", vocab["relation"])
        foreground: _enum("foreground", vocab["foreground"])
        background: _enum("background", vocab["background"])

    return Output


def pick(value: object, choices: list[str]) -> str:
    text = str(value).strip()
    if text in choices:
        return text
    for part in text.replace(",", " ").split():
        if part in choices:
            return part
    return ""


def blank() -> dict[str, str]:
    return {key: "" for key in KEYS}


def image_b64(path: Path) -> str:
    import base64

    return base64.b64encode(path.read_bytes()).decode()


def post_json(url: str, body: dict, headers: dict | None = None, timeout: float = 60) -> dict:
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdrs)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())
