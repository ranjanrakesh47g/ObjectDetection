"""Label an uploaded image or video with Qwen 27B on OpenRouter."""

import json
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from src.utils.base import DATA
from src.utils.metrics import vocab_for
from src.utils.openrouter import OpenRouter
from src.utils.video import label_clip

VIDEOS = {".mp4", ".mov", ".webm", ".mkv"}
app = FastAPI()
model = OpenRouter("qwen/qwen3.8-27b", "qwen3.8-27b_openrouter")


def _vocab() -> dict[str, list[str]]:
    rows = []
    for split in ("ecommerce", "news"):
        for name in ("gt_images.json", "gt_videos.json"):
            path = DATA / split / name
            if path.exists():
                rows += json.loads(path.read_text())
    return vocab_for(rows)


VOCAB = _vocab()


@app.get("/health_check")
def health_check() -> dict:
    return {"status": "ok"}


def _jpeg(path: Path) -> Path:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return path
    dest = path.with_suffix(".jpg")
    Image.open(path).convert("RGB").save(dest)
    return dest


@app.post("/detect")
async def detect(file: UploadFile) -> dict:
    suffix = Path(file.filename or "upload").suffix.lower()
    video = suffix in VIDEOS or (file.content_type or "").startswith("video/")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix or '.bin'}"
        path.write_bytes(await file.read())
        try:
            pred = label_clip(model, path, VOCAB)[0] if video else model.predict(_jpeg(path), VOCAB)
        except Exception as err:
            raise HTTPException(400, str(err)) from err
    pred["latency"] = f"{time.perf_counter() - started:.3f} sec"
    return pred
