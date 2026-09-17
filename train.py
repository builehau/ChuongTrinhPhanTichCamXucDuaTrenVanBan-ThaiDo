from __future__ import annotations

import argparse
from pathlib import Path

from src.training import train_and_evaluate


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình cảm xúc tiếng Việt.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sentiment_vi.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "sentiment_model.joblib")
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        data_path=args.data,
        model_path=args.model,
        metrics_path=args.metrics,
        confusion_matrix_path=ROOT / "results" / "confusion_matrix.png",
        random_state=args.seed,
    )
    print("Huấn luyện hoàn tất!")
    print(f"Số mẫu: {metrics['dataset_size']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(
        "5-fold CV Macro F1: "
        f"{metrics['cv_macro_f1_mean']:.4f} ± {metrics['cv_macro_f1_std']:.4f}"
    )
    print(f"Mô hình: {args.model}")


if __name__ == "__main__":
    main()

