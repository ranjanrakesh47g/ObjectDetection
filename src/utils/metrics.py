"""Write per-image predictions and the accuracy tables."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.utils.base import DATA, KEYS, RESULTS

COST = {
    "qwen3.8-27b_openrouter": 0.00016,
    "qwen3-vl-30b-a3b_openrouter": 0.000279,
    "gemini-3.1-flash-lite_google": 0.000485,
}
FIELDS = (("fg", "foreground"), ("bg", "background"), ("rel", "relation"))
SPLITS = (("ecommerce", "ecom"), ("news", "news"))
_STATS = ("acc", "micro_f1", "macro_f1")
METRIC_COLUMNS = ["model_name"]
METRIC_COLUMNS += [
    f"{prefix}_{short}_{stat}"
    for _, prefix in SPLITS
    for short, _key in FIELDS
    for stat in _STATS
]
METRIC_COLUMNS += [f"{prefix}_{stat}" for _, prefix in SPLITS for stat in _STATS]
METRIC_COLUMNS += ["acc", "micro_f1", "macro_f1", "latency (s)"]


def _f1(gold, pred, average: str) -> float:
    labels = sorted(set(gold))
    return round(f1_score(gold, pred, average=average, labels=labels, zero_division=0), 3)


def _ground_truth() -> pd.DataFrame:
    rows = []
    for split, _ in SPLITS:
        for row in json.loads((DATA / split / "gt_images.json").read_text()):
            rows.append({
                "image_filepath": f"data/{split}/images/{row['file']}",
                "gt_fg": row["foreground"],
                "gt_bg": row["background"],
                "gt_rel": row["relation"],
            })
    return pd.DataFrame(rows).set_index("image_filepath")


def _metrics_row(model: str, frame: pd.DataFrame) -> dict:
    row = {"model_name": model}
    total = 0
    all_gold, all_pred = [], []
    for split, prefix in SPLITS:
        part = frame[frame.image_filepath.str.startswith(f"data/{split}/")]
        correct, golds, preds = 0, [], []
        for short, _key in FIELDS:
            gold, pred = part[f"gt_{short}"], part[f"{model}_{short}"].fillna("")
            correct += int(gold.eq(pred).sum())
            row[f"{prefix}_{short}_acc"] = round(accuracy_score(gold, pred), 3)
            row[f"{prefix}_{short}_micro_f1"] = _f1(gold, pred, "micro")
            row[f"{prefix}_{short}_macro_f1"] = _f1(gold, pred, "macro")
            golds += [f"{short}:{value}" for value in gold]
            preds += [f"{short}:{value}" for value in pred]
        row[f"{prefix}_acc"] = round(correct / (len(part) * 3), 3)
        row[f"{prefix}_micro_f1"] = _f1(golds, preds, "micro")
        row[f"{prefix}_macro_f1"] = _f1(golds, preds, "macro")
        total += correct
        all_gold += golds
        all_pred += preds
    row["acc"] = round(total / (len(frame) * 3), 3)
    row["micro_f1"] = _f1(all_gold, all_pred, "micro")
    row["macro_f1"] = _f1(all_gold, all_pred, "macro")
    row["latency (s)"] = round(frame[f"{model}_latency"].mean(), 3)
    return row


def _source(name: str) -> str:
    if "openrouter" in name or "google" in name:
        return "commercial"
    return "open-source"


def save_metrics(model: str, results: Path = RESULTS) -> None:
    frame = pd.read_csv(results / "predictions.csv").dropna(subset=[f"{model}_latency"])
    row = _metrics_row(model, frame)
    path = results / "metrics.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=METRIC_COLUMNS)
    hit = old.model_name == model
    if old.empty:
        old = pd.DataFrame([row])
    elif hit.any():
        old.loc[hit, list(row)] = list(row.values())
    else:
        old = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    old = old.reindex(columns=METRIC_COLUMNS)
    old.to_csv(path, index=False)

    out = old[["model_name", "acc", "micro_f1", "macro_f1", "latency (s)"]].copy()
    out.insert(1, "source", [_source(name) for name in out.model_name])
    out["cost ($/1k images)"] = [round(COST.get(name, 0) * 1000, 3) for name in out.model_name]
    out.to_csv(results / "overall_metrics.csv", index=False)


def save_predictions(model: str, frame: pd.DataFrame, results: Path = RESULTS) -> None:
    """Merge one model's labels into predictions.csv, then refresh metrics."""
    results.mkdir(exist_ok=True)
    frame = frame.rename(columns={
        "image": "image_filepath", "foreground": "fg", "background": "bg", "relation": "rel",
    }).set_index("image_filepath")
    frame.columns = [f"{model}_{column}" for column in frame.columns]
    path = results / "predictions.csv"
    parts = [_ground_truth(), frame]
    if path.exists():
        old = pd.read_csv(path).set_index("image_filepath")
        drop = [column for column in old.columns if column in frame.columns or column.startswith("gt_")]
        parts = [_ground_truth(), old.drop(columns=drop), frame]
    pd.concat(parts, axis=1).to_csv(path)
    save_metrics(model, results)


def vocab_for(rows: list[dict]) -> dict[str, list[str]]:
    return {key: sorted({row[key] for row in rows}) for key in KEYS}
