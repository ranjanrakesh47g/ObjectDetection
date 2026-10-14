"""Load one detector, label each image, then write metrics."""

import argparse
import json
import time
from collections.abc import Callable

import pandas as pd
from tqdm import tqdm

from src.utils.base import DATA, Detector
from src.utils.gemini import Gemini
from src.utils.metrics import SPLITS, save_predictions, vocab_for
from src.utils.openrouter import OpenRouter
from src.utils.qwen import Qwen

FACTORIES: dict[str, Callable[[], Detector]] = {
    "qwen3.8-27b_openrouter": lambda: OpenRouter("qwen/qwen3.8-27b", "qwen3.8-27b_openrouter"),
    "qwen3-vl-30b-a3b_openrouter": lambda: OpenRouter("qwen/qwen3-vl-30b-a3b-instruct", "qwen3-vl-30b-a3b_openrouter"),
    "gemini-3.1-flash-lite_google": Gemini,
    "qwen3-vl-2b_4bit_local": lambda: Qwen("Qwen/Qwen3-VL-2B-Instruct", "qwen3-vl-2b_4bit_local", bits=4),
    "qwen3-vl-4b_colab": lambda: Qwen("Qwen/Qwen3-VL-4B-Instruct", "qwen3-vl-4b_colab", tools=True),
    "qwen3-vl-8b_4bit_colab": lambda: Qwen("Qwen/Qwen3-VL-8B-Instruct", "qwen3-vl-8b_4bit_colab", bits=4, tools=True),
}


def evaluate(model: Detector) -> None:
    """Predict every image with the already loaded model, then save."""
    logged = []
    for split, _prefix in SPLITS:
        rows = json.loads((DATA / split / "gt_images.json").read_text())
        vocab = vocab_for(rows)
        for row in tqdm(rows, desc=f"{model.name} {split}"):
            path = DATA / split / "images" / row["file"]
            started = time.perf_counter()
            pred = model.predict(path, vocab)
            latency = round(time.perf_counter() - started, 3)
            logged.append({
                "image": f"data/{split}/images/{row['file']}",
                "foreground": pred["foreground"],
                "background": pred["background"],
                "relation": pred["relation"],
                "latency": latency,
            })
            tqdm.write(f"{split} {row['file']} {pred} {latency:.2f}s")
            if model.pause:
                time.sleep(model.pause)
    frame = pd.DataFrame(logged)
    print(f"{frame['latency'].mean():.2f}s/image")
    save_predictions(model.name, frame)


def run(name: str) -> None:
    """Construct one model, score the dataset, then free it."""
    if name not in FACTORIES:
        known = ", ".join(FACTORIES)
        raise ValueError(f"Unknown model {name}. Choose from: {known}")
    model = FACTORIES[name]()
    try:
        evaluate(model)
    finally:
        model.close()


def run_all() -> None:
    for name in FACTORIES:
        run(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.utils",
        description="Load one model, label each image, save metrics.",
    )
    parser.add_argument("name", nargs="?", choices=list(FACTORIES), help="model to run")
    parser.add_argument("--all", action="store_true", help="run every model, one at a time")
    args = parser.parse_args()
    if args.all:
        run_all()
    elif args.name:
        run(args.name)
    else:
        parser.error("pass a model name, or --all")


if __name__ == "__main__":
    main()
