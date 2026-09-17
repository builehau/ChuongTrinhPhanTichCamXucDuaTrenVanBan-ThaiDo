from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sentiment_model import build_pipeline, normalize_text, predict_one  # noqa: E402
from src.training import load_and_validate_dataset  # noqa: E402


class SentimentModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_and_validate_dataset(ROOT / "data" / "sentiment_vi.csv")
        cls.model = build_pipeline()
        cls.model.fit(cls.data["text"], cls.data["label"])
        cls.artifact = {"model": cls.model, "metadata": {}}

    def test_dataset_is_balanced(self) -> None:
        counts = self.data["label"].value_counts()
        self.assertEqual(counts.nunique(), 1)
        self.assertGreaterEqual(counts.min(), 50)

    def test_normalization(self) -> None:
        self.assertEqual(normalize_text("  RẤT   TỐT!!!  "), "rất tốt!")
        self.assertIn("tốt", normalize_text("Sản phẩm 👍"))

    def test_positive_prediction(self) -> None:
        result = predict_one(self.artifact, "Sản phẩm tuyệt vời, tôi rất hài lòng")
        self.assertEqual(result["label"], "tich_cuc")

    def test_neutral_prediction(self) -> None:
        result = predict_one(self.artifact, "Cuộc họp diễn ra vào thứ hai tuần tới")
        self.assertEqual(result["label"], "trung_lap")

    def test_negative_prediction(self) -> None:
        result = predict_one(self.artifact, "Dịch vụ quá tệ và rất chậm")
        self.assertEqual(result["label"], "tieu_cuc")

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            predict_one(self.artifact, "   ")


if __name__ == "__main__":
    unittest.main()

