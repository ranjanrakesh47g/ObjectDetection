"""One-image detectors and the loop that scores them."""

from src.utils.evaluate import evaluate, run, run_all
from src.utils.gemini import Gemini
from src.utils.openrouter import OpenRouter
from src.utils.qwen import Qwen
from src.utils.vision import CloudVision

__all__ = [
    "CloudVision",
    "Gemini",
    "OpenRouter",
    "Qwen",
    "evaluate",
    "run",
    "run_all",
]
