from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sentiment_model import LABEL_INFO, load_artifact, predict_one


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc của văn bản tiếng Việt.")
    parser.add_argument("--text", type=str, help="Văn bản cần phân tích")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "sentiment_model.joblib")
    parser.add_argument("--json", action="store_true", help="In kết quả dạng JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit("Chưa có mô hình. Hãy chạy: python train.py")
    text = args.text or input("Nhập văn bản cần phân tích: ").strip()
    result = predict_one(load_artifact(args.model), text)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"Kết quả: {result['emoji']} {result['name']}")
    print(f"Độ tin cậy: {result['confidence']:.2%}")
    print("Xác suất từng lớp:")
    for label, probability in sorted(result["probabilities"].items(), key=lambda item: -item[1]):
        print(f"  - {LABEL_INFO[label]['name']}: {probability:.2%}")
    if result["evidence"]:
        print("Từ/cụm từ ảnh hưởng:", ", ".join(result["evidence"]))


if __name__ == "__main__":
    main()

