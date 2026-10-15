"""Label a clip from the clearest face in each shot, then majority-vote."""

import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.base import DATA, KEYS, MODELS, RESULTS, Detector
from src.utils.metrics import COST, _f1, _source, vocab_for

SCENE = 0.3
GAP = 0.5


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout)


def _cuts(path: Path) -> list[float]:
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-vf", "select='gte(scene,0)',metadata=print", "-an", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    cuts, stamp = [], None
    for line in result.stderr.splitlines():
        if "pts_time:" in line:
            stamp = float(line.rsplit("pts_time:", 1)[-1])
        elif stamp is not None and "lavfi.scene_score=" in line:
            if float(line.rsplit("=", 1)[-1]) > SCENE:
                cuts.append(stamp)
            stamp = None
    return cuts


def _shots(path: Path) -> list[tuple[float, float]]:
    duration = _duration(path)
    marks = [0.0]
    for cut in _cuts(path):
        if cut - marks[-1] >= GAP and duration - cut >= GAP:
            marks.append(cut)
    marks.append(duration)
    return list(zip(marks, marks[1:]))


def _times(path: Path) -> list[float]:
    """Largest face in each shot. A shot with no face keeps its midpoint."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(fps / 5)), 1)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    model = MODELS / "face_detection_yunet_2023mar.onnx"
    detector = cv2.FaceDetectorYN.create(str(model), "", size, 0.6, 0.3, 500)
    times = []
    for left, right in _shots(path):
        best_area, best_t, best_score = 0, (left + right) / 2, 0.0
        index = int(left * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        while index < int(right * fps):
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                _count, faces = detector.detect(frame)
                if faces is not None:
                    for face in faces:
                        if min(face[2], face[3]) < 24:
                            continue
                        area = int(face[2] * face[3])
                        if area > best_area:
                            best_area, best_t, best_score = area, index / fps, float(face[-1])
            index += 1
        times.append((best_t, best_score))
    cap.release()
    return times


def keyframes(path: Path, directory: Path) -> list[tuple[Path, float]]:
    frames = []
    for index, (stamp, score) in enumerate(_times(path)):
        frame = directory / f"{index:02d}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ss", f"{stamp:.3f}", "-frames:v", "1", str(frame)],
            check=True, capture_output=True,
        )
        frames.append((frame, score))
    return frames


def vote(preds: list[dict[str, str]], scores: list[float]) -> dict[str, str]:
    """Most common full label. A tie keeps the frame with the higher face score."""
    counts = Counter(tuple(pred[key] for key in KEYS) for pred in preds)
    best = max(counts.values())
    chosen, chosen_score = preds[0], -1.0
    for pred, score in zip(preds, scores):
        if counts[tuple(pred[key] for key in KEYS)] == best and score > chosen_score:
            chosen, chosen_score = pred, score
    return {key: chosen[key] for key in KEYS}


def label_clip(model: Detector, path: Path, vocab: dict[str, list[str]]) -> tuple[dict[str, str], list[dict[str, str]]]:
    with tempfile.TemporaryDirectory() as tmp:
        frames = keyframes(path, Path(tmp))
        preds = []
        for index, (frame, _score) in enumerate(frames):
            if index and model.pause:
                time.sleep(model.pause)
            preds.append(model.predict(frame, vocab))
        return vote(preds, [score for _frame, score in frames]), preds


def _upsert(path: Path, frame: pd.DataFrame, name: str) -> None:
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old[old.model_name != name], frame], ignore_index=True)
    frame.to_csv(path, index=False)


_PREDICT_BASE = ["video_filepath", "key_frames", "gt_fg", "gt_bg", "gt_rel"]


def _wide(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "video_filepath": "data/news/videos/" + frame.video,
        "key_frames": frame.frames,
        "gt_fg": frame.gt_foreground,
        "gt_bg": frame.gt_background,
        "gt_rel": frame.gt_relation,
        f"{name}_fg": frame.foreground,
        f"{name}_bg": frame.background,
        f"{name}_rel": frame.relation,
        f"{name}_latency": frame.latency,
    })


def _metrics(name: str, frame: pd.DataFrame) -> dict:
    row = {"model_name": name}
    correct, golds, preds = 0, [], []
    for key, short in zip(KEYS, ("fg", "bg", "rel")):
        gold, pred = frame[f"gt_{key}"], frame[key]
        hits = int(gold.eq(pred).sum())
        row[f"{short}_acc"] = round(hits / len(frame), 3)
        row[f"{short}_micro_f1"] = _f1(gold, pred, "micro")
        row[f"{short}_macro_f1"] = _f1(gold, pred, "macro")
        correct += hits
        golds += [f"{short}:{value}" for value in gold]
        preds += [f"{short}:{value}" for value in pred]
    row["acc"] = round(correct / (len(frame) * len(KEYS)), 3)
    row["micro_f1"] = _f1(golds, preds, "micro")
    row["macro_f1"] = _f1(golds, preds, "macro")
    row["latency (s)"] = round(float(frame.latency.mean()), 3)
    return row


def _save(name: str, frame: pd.DataFrame) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "video_predictions.csv"
    wide = _wide(name, frame)
    if path.exists():
        old = pd.read_csv(path)
        drop = [column for column in old.columns if column.startswith(f"{name}_")]
        old = old.drop(columns=drop)
        fresh = ["key_frames", "gt_fg", "gt_bg", "gt_rel"]
        wide = old.drop(columns=[column for column in fresh if column in old.columns]).merge(
            wide, on="video_filepath", how="outer",
        )
    columns = _PREDICT_BASE + [column for column in wide.columns if column not in _PREDICT_BASE]
    wide[columns].to_csv(path, index=False)
    _upsert(RESULTS / "video_metrics.csv", pd.DataFrame([_metrics(name, frame)]), name)
    _overall(pd.read_csv(RESULTS / "video_metrics.csv"))


def _overall(metrics: pd.DataFrame) -> None:
    out = metrics[["model_name", "acc", "micro_f1", "macro_f1", "latency (s)"]].copy()
    out.insert(1, "source", [_source(name) for name in out.model_name])
    out["cost ($/1k keyframes)"] = [round(COST.get(name, 0) * 1000, 3) for name in out.model_name]
    out.to_csv(RESULTS / "video_overall_metrics.csv", index=False)


def label_videos(model: Detector, split: str = "news") -> pd.DataFrame:
    """Label each video in one split and save the video result files."""
    rows = json.loads((DATA / split / "gt_videos.json").read_text())
    vocab = vocab_for(rows)
    logged = []
    for index, row in enumerate(tqdm(rows, desc=f"{model.name} {split} video")):
        if index and model.pause:
            time.sleep(model.pause)
        path = DATA / split / "videos" / row["video"]
        started = time.perf_counter()
        pred, preds = label_clip(model, path, vocab)
        latency = round(time.perf_counter() - started, 3)
        logged.append({
            "video": row["video"],
            "frames": len(preds),
            "foreground": pred["foreground"],
            "gt_foreground": row["foreground"],
            "background": pred["background"],
            "gt_background": row["background"],
            "relation": pred["relation"],
            "gt_relation": row["relation"],
            "latency": latency,
        })
        detail = " | ".join(str(item) for item in preds)
        tqdm.write(f"{row['video']} n={len(preds)} {pred} {latency:.2f}s [{detail}]")
    frame = pd.DataFrame(logged)
    _save(model.name, frame)
    print(f"{_metrics(model.name, frame)['acc']:.3f} acc, {frame['latency'].mean():.2f}s/clip")
    return frame


def run_videos(name: str) -> None:
    """Construct one model, label each news clip, then free it."""
    from src.utils.evaluate import FACTORIES

    if name not in FACTORIES:
        known = ", ".join(FACTORIES)
        raise ValueError(f"Unknown model {name}. Choose from: {known}")
    model = FACTORIES[name]()
    try:
        label_videos(model)
    finally:
        model.close()
