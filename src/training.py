from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .sentiment_model import LABELS, build_pipeline, normalize_text, save_artifact


def load_and_validate_dataset(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {data_path}")

    data = pd.read_csv(data_path)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("CSV phải có đúng hai cột bắt buộc: text, label")

    data = data[["text", "label"]].dropna().copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()
    data = data[data["text"] != ""]

    invalid_labels = sorted(set(data["label"]) - set(LABELS))
    if invalid_labels:
        raise ValueError(f"Nhãn không hợp lệ: {invalid_labels}. Chỉ dùng: {LABELS}")

    data["normalized"] = data["text"].map(normalize_text)
    conflicts = data.groupby("normalized")["label"].nunique()
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index.tolist()[:3]
        raise ValueError(f"Có văn bản trùng nhưng khác nhãn: {examples}")
    data = data.drop_duplicates(subset=["normalized"]).drop(columns="normalized")

    counts = data["label"].value_counts()
    missing = [label for label in LABELS if counts.get(label, 0) < 10]
    if missing:
        raise ValueError(f"Mỗi lớp cần ít nhất 10 mẫu. Lớp chưa đủ: {missing}")
    return data.reset_index(drop=True)


def _save_confusion_matrix(matrix: list[list[int]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    display = ["Tiêu cực", "Trung lập", "Tích cực"]
    ax.set(xticks=range(3), yticks=range(3), xticklabels=display, yticklabels=display)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title("Ma trận nhầm lẫn")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, matrix[row][col], ha="center", va="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def train_and_evaluate(
    data_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    confusion_matrix_path: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    data = load_and_validate_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        random_state=random_state,
        stratify=data["label"],
    )

    model = build_pipeline(random_state=random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    matrix = confusion_matrix(y_test, predictions, labels=LABELS).tolist()
    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_model = build_pipeline(random_state=random_state)
    cv_scores = cross_val_score(cv_model, data["text"], data["label"], cv=cv, scoring="f1_macro")

    metrics: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "TF-IDF (word + character n-grams) + lexicon features + Logistic Regression",
        "random_state": random_state,
        "dataset_size": int(len(data)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "class_distribution": {key: int(value) for key, value in data["label"].value_counts().to_dict().items()},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": matrix,
        "classification_report": report,
    }

    metadata = {
        "labels": LABELS,
        "dataset_size": int(len(data)),
        "algorithm": metrics["algorithm"],
        "created_at_utc": metrics["created_at_utc"],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        },
    }
    save_artifact(model, model_path, metadata)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if confusion_matrix_path is not None:
        _save_confusion_matrix(matrix, Path(confusion_matrix_path))
    return metrics
