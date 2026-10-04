from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(r"C:\Users\91819\Desktop\da")
LOCAL_ULTRALYTICS_CONFIG = ROOT / ".ultralytics"
LOCAL_ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_ULTRALYTICS_CONFIG))

from ultralytics import YOLO


DATA = ROOT / "data_ours_poisson.yaml"
MODEL = ROOT / "yolov8n.pt"
PROJECT = ROOT / "runs" / "detect"
SEEDS = (42, 3407)


def write_test_summary(seed: int, metrics) -> None:
    box = metrics.box
    row = {
        "seed": seed,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }
    out = PROJECT / f"Ours_poisson_seed{seed}_test" / "test_metrics.json"
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    combined = PROJECT / "Ours_poisson_multiseed_test_summary.csv"
    existing = []
    if combined.exists():
        with combined.open("r", newline="", encoding="utf-8-sig") as f:
            existing = [r for r in csv.DictReader(f) if int(r["seed"]) != seed]
    existing.append({key: str(value) for key, value in row.items()})
    existing.sort(key=lambda r: int(r["seed"]))
    with combined.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    for seed in SEEDS:
        train_name = f"Ours_poisson_seed{seed}"
        train_dir = PROJECT / train_name
        if train_dir.exists():
            raise FileExistsError(
                f"结果目录已存在，为避免覆盖而停止：{train_dir}"
            )

        model = YOLO(str(MODEL))
        model.train(
            data=str(DATA),
            epochs=300,
            patience=50,
            batch=4,
            imgsz=640,
            project=str(PROJECT),
            name=train_name,
            workers=2,
            seed=seed,
            deterministic=True,
        )

        best = train_dir / "weights" / "best.pt"
        test_model = YOLO(str(best))
        test_metrics = test_model.val(
            data=str(DATA),
            split="test",
            project=str(PROJECT),
            name=f"{train_name}_test",
            seed=seed,
            deterministic=True,
        )
        write_test_summary(seed, test_metrics)


if __name__ == "__main__":
    main()
