from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


LABELS = ["tieu_cuc", "trung_lap", "tich_cuc"]
LABEL_INFO = {
    "tich_cuc": {"name": "Tích cực", "emoji": "😊", "score": 1},
    "trung_lap": {"name": "Trung lập", "emoji": "😐", "score": 0},
    "tieu_cuc": {"name": "Tiêu cực", "emoji": "😞", "score": -1},
}

EMOJI_MAP = {
    "😍": " rất thích ",
    "🥰": " yêu thích ",
    "😊": " vui hài lòng ",
    "😀": " vui ",
    "👍": " tốt ",
    "❤️": " yêu thích ",
    "❤": " yêu thích ",
    "😂": " buồn cười ",
    "😐": " bình thường ",
    "😕": " khó hiểu ",
    "😞": " thất vọng ",
    "😢": " buồn ",
    "😭": " rất buồn ",
    "😡": " tức giận ",
    "👎": " tệ ",
}

POSITIVE_TERMS = (
    "tốt", "tuyệt vời", "xuất sắc", "hài lòng", "yêu thích", "thích", "đẹp",
    "ngon", "nhanh", "thân thiện", "dễ sử dụng", "dễ hiểu", "hữu ích", "ổn",
    "hiệu quả", "chuyên nghiệp", "đáng tiền", "vui", "an toàn", "hợp lý", "sạch",
    "thuận tiện", "mượt", "tận tâm", "chu đáo", "nhiệt tình", "sáng tạo", "tin tưởng",
    "xịn", "ủng hộ", "cuốn hút", "hấp dẫn", "thoải mái", "rõ ràng", "thỏa đáng",
    "đáng khen", "động lực", "cầu thị", "chính xác", "sớm", "tươi", "vừa miệng",
    "không thất vọng", "không hề thất vọng","tuyệt","vượt mong đợi","tuyệt vời","tuyệt hảo","tuyệt cú mèo","tuyệt đỉnh","tuyệt phẩm","tuyệt vời nhất","tuyệt vời quá","tuyệt vời lắm","tuyệt vời quá trời","tuyệt vời quá đi","tuyệt vời quá trời ơi","tuyệt vời quá trời ơi luôn","tuyệt vời quá trời ơi luôn luôn","tuyệt vời quá trời ơi luôn luôn luôn"
"vượt kỳ vọng","vượt mong đợi","vượt trội","vượt bậc","vượt xa","vượt lên","vượt qua","vượt trội hơn","vượt bậc hơn","vượt xa hơn","vượt lên hơn","vượt qua hơn"
)
NEGATIVE_TERMS = (
    "tệ", "thất vọng", "bị lỗi", "báo lỗi", "chậm", "kém", "không ngon",
    "không hài lòng", "không đồng ý", "khó hiểu", "khó dùng", "bẩn", "ồn",
    "khó chịu", "rè", "quá cao", "buồn", "chật", "rườm rà", "yếu", "nóng",
    "hời hợt", "muộn", "xấu", "chập chờn", "dở", "lộn xộn", "thờ ơ", "không đáng",
    "trễ", "mờ", "sai", "mất dữ liệu", "thiếu", "nghiêm trọng", "chán", "hàng giả",
    "bị vỡ", "phức tạp", "mệt mỏi", "phớt lờ", "nhạt", "gượng gạo", "cẩu thả",
    "bất tiện", "bị ngắt", "không tin tưởng", "không quay lại", "hoàn tiền", 
    "không hài lòng","không tốt", "không ổn", "không vui", "không đẹp",
      "không ngon", "không nhanh","không thân thiện", "không dễ sử dụng", "không dễ hiểu", "không hữu ích",
      "không ổn", "không hiệu quả", "không chuyên nghiệp", "không đáng tiền", "không vui", "không an toàn", 
      "không hợp lý", "không sạch", "không thuận tiện", "không mượt",
      "không","không thích","chán","xấu","hại"
)


def normalize_text(text: str) -> str:
    """Chuẩn hóa nhẹ, vẫn giữ từ phủ định và dấu tiếng Việt."""
    text = unicodedata.normalize("NFC", html.unescape(str(text))).lower().strip()
    for emoji, meaning in EMOJI_MAP.items():
        text = text.replace(emoji, meaning)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", " EMAIL ", text)
    text = re.sub(r"\b\d+(?:[.,]\d+)?\b", " NUM ", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Trích xuất tín hiệu cảm xúc thủ công để bổ sung cho TF-IDF."""

    def fit(self, x, y=None):  # noqa: D401, ANN001
        return self

    def transform(self, texts):  # noqa: ANN001
        rows = []
        for text in texts:
            normalized = normalize_text(text)
            positive = sum(normalized.count(term) for term in POSITIVE_TERMS)
            negative = sum(normalized.count(term) for term in NEGATIVE_TERMS)
            # Một câu có "nhưng/tuy nhiên" thường nhấn mạnh mệnh đề đứng sau.
            contrast = int(" nhưng " in f" {normalized} " or "tuy nhiên" in normalized)
            rows.append([positive, negative, positive - negative, contrast])
        return csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        return np.asarray(["positive_count", "negative_count", "sentiment_balance", "has_contrast"])


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Tạo pipeline TF-IDF từ + ký tự, sau đó phân lớp Logistic Regression."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    analyzer="word",
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w+\b",
                    sublinear_tf=True,
                    max_features=25_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    max_features=30_000,
                ),
            ),
            ("lexicon", LexiconFeatures()),
        ],
        transformer_weights={"word": 1.0, "char": 0.55, "lexicon": 0.5},
    )
    classifier = LogisticRegression(
        C=0.5,
        max_iter=2_000,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def save_artifact(model: Pipeline, path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Tệp mô hình không đúng định dạng của dự án.")
    return artifact


def explain_prediction(model: Pipeline, text: str, label: str, top_n: int = 5) -> list[str]:
    """Lấy các đặc trưng có đóng góp dương lớn nhất cho lớp dự đoán."""
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    vector = features.transform([text])
    class_index = int(np.where(classifier.classes_ == label)[0][0])
    contributions = vector.multiply(classifier.coef_[class_index]).toarray()[0]
    names = features.get_feature_names_out()
    ranked = np.argsort(contributions)[::-1]

    evidence: list[str] = []
    for index in ranked:
        if contributions[index] <= 0:
            break
        feature = str(names[index])
        if feature.startswith("word__"):
            clean = feature.removeprefix("word__")
            if clean not in evidence:
                evidence.append(clean)
        if len(evidence) >= top_n:
            break
    return evidence


def predict_one(artifact: dict[str, Any], text: str) -> dict[str, Any]:
    if not str(text).strip():
        raise ValueError("Văn bản không được để trống.")
    model: Pipeline = artifact["model"]
    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_
    best_index = int(np.argmax(probabilities))
    label = str(classes[best_index])
    info = LABEL_INFO[label]
    probability_map = {str(c): float(p) for c, p in zip(classes, probabilities)}
    return {
        "label": label,
        "name": info["name"],
        "emoji": info["emoji"],
        "score": info["score"],
        "confidence": float(probabilities[best_index]),
        "probabilities": probability_map,
        "evidence": explain_prediction(model, text, label),
        "normalized_text": normalize_text(text),
    }


def predict_many(artifact: dict[str, Any], texts: list[str]) -> list[dict[str, Any]]:
    return [predict_one(artifact, text) for text in texts]
