"""Shrink photos and play the short clips inside the notebook."""

import html
import json

import ipyplot
import numpy as np
from IPython.display import Video, display
from PIL import Image

from src.data.config import ROOT
from src.data.media import run_ffmpeg


def _caption(row: dict) -> str:
    return " | ".join(f"{key}: {row.get(key, '')}" for key in ("relation", "foreground", "background"))


def mark(path, side: int = 160) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image.thumbnail((side, side))
    return np.asarray(image)


def show(name: str, n: int | None = None, side: int = 160, gt: str = "gt_images.json") -> None:
    rows = json.loads((ROOT / name / gt).read_text())
    if n:
        rows = rows[:n]
    images = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        images[index] = mark(ROOT / name / "images" / row["file"], side)
    labels = []
    for row in rows:
        text = html.escape(_caption(row))
        labels.append(f'<b style="color:#000;background:#fff;font-weight:normal">{text}</b>')
    ipyplot.plot_images(images, labels, img_width=side)


def play(name: str, n: int | None = None, width: int = 320, gt: str = "gt_videos.json") -> None:
    rows = json.loads((ROOT / name / gt).read_text())
    if n:
        rows = rows[:n]
    for row in rows:
        source = ROOT / name / "videos" / row["video"]
        preview = source.with_name("small_" + source.name)
        if not preview.exists():
            run_ffmpeg(
                "-i",
                str(source),
                "-vf",
                f"scale={width}:-2",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "32",
                str(preview),
            )
        print(_caption(row))
        display(Video(str(preview), embed=True, width=width))


def counts():
    import pandas as pd

    records = []
    for name in ("ecommerce", "news"):
        images = json.loads((ROOT / name / "gt_images.json").read_text())
        videos = ROOT / name / "gt_videos.json"
        labeled = images + (json.loads(videos.read_text()) if videos.exists() else [])
        records.append({
            "images": len(images),
            "#foreground_classes": len({row["foreground"] for row in labeled}),
            "#background_classes": len({row["background"] for row in labeled}),
        })
    return pd.DataFrame(records, index=["ecommerce", "news"])
