from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.sentiment_model import LABEL_INFO, load_artifact, predict_one
from src.training import train_and_evaluate


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
DATA_PATH = ROOT / "data" / "sentiment_vi.csv"
METRICS_PATH = ROOT / "results" / "metrics.json"

st.set_page_config(page_title="Phân tích cảm xúc tiếng Việt", page_icon="💬", layout="wide")


@st.cache_resource
def get_artifact():
    if not MODEL_PATH.exists():
        with st.spinner("Đang huấn luyện mô hình lần đầu..."):
            train_and_evaluate(
                DATA_PATH,
                MODEL_PATH,
                METRICS_PATH,
                ROOT / "results" / "confusion_matrix.png",
            )
    return load_artifact(MODEL_PATH)


def probability_frame(result: dict) -> pd.DataFrame:
    rows = [
        {"Cảm xúc": LABEL_INFO[label]["name"], "Xác suất": probability}
        for label, probability in result["probabilities"].items()
    ]
    return pd.DataFrame(rows).set_index("Cảm xúc")


st.title("💬 Phân tích cảm xúc / thái độ từ văn bản")
st.caption("Mô hình TF-IDF + Logistic Regression · 3 lớp: tích cực, trung lập, tiêu cực")

tab_single, tab_batch, tab_model = st.tabs(["Phân tích một câu", "Phân tích tệp CSV", "Thông tin mô hình"])

with tab_single:
    examples = {
        "Tích cực": "Dịch vụ rất nhanh và nhân viên cực kỳ nhiệt tình!",
        "Trung lập": "Cuộc họp sẽ bắt đầu lúc 9 giờ sáng mai.",
        "Tiêu cực": "Ứng dụng quá chậm và liên tục bị lỗi.",
        "Câu có 'nhưng'": "Giao hàng hơi chậm nhưng sản phẩm dùng rất tốt.",
    }
    selected_example = st.selectbox("Chọn câu mẫu (không bắt buộc)", ["Tự nhập"] + list(examples))
    default_text = "" if selected_example == "Tự nhập" else examples[selected_example]
    text = st.text_area("Nhập bình luận hoặc câu tiếng Việt", value=default_text, height=140)

    if st.button("Phân tích cảm xúc", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Bạn hãy nhập một câu trước khi phân tích.")
        else:
            result = predict_one(get_artifact(), text)
            left, right = st.columns([1, 2])
            with left:
                st.metric("Kết quả", f"{result['emoji']} {result['name']}")
                st.metric("Độ tin cậy", f"{result['confidence']:.2%}")
                if result["evidence"]:
                    st.write("Từ/cụm từ ảnh hưởng:", ", ".join(result["evidence"]))
            with right:
                st.bar_chart(probability_frame(result), y="Xác suất")
            if result["confidence"] < 0.55:
                st.info("Mô hình chưa thật sự chắc chắn. Câu có thể mơ hồ hoặc khác dữ liệu huấn luyện.")

with tab_batch:
    st.write("Tải lên tệp CSV có một cột chứa văn bản. Kết quả có thể tải xuống ngay sau khi phân tích.")
    uploaded = st.file_uploader("Chọn tệp CSV", type=["csv"])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Không đọc được CSV: {exc}")
        else:
            if frame.empty:
                st.warning("Tệp CSV không có dữ liệu.")
            else:
                text_column = st.selectbox("Cột chứa văn bản", list(frame.columns))
                if st.button("Phân tích toàn bộ tệp"):
                    valid = frame[text_column].fillna("").astype(str)
                    results = []
                    for value in valid:
                        if value.strip():
                            results.append(predict_one(get_artifact(), value))
                        else:
                            results.append({"label": "", "name": "", "confidence": 0.0, "score": 0})
                    output = frame.copy()
                    output["sentiment_label"] = [item["label"] for item in results]
                    output["sentiment_name"] = [item["name"] for item in results]
                    output["sentiment_score"] = [item["score"] for item in results]
                    output["confidence"] = [round(item["confidence"], 4) for item in results]
                    st.dataframe(output, use_container_width=True)
                    st.download_button(
                        "Tải kết quả CSV",
                        output.to_csv(index=False).encode("utf-8-sig"),
                        file_name="ket_qua_cam_xuc.csv",
                        mime="text/csv",
                    )

with tab_model:
    artifact = get_artifact()
    metadata = artifact.get("metadata", {})
    st.subheader("Cấu hình")
    st.json(metadata)
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        col1, col2, col3 = st.columns(3)
        col1.metric("Số mẫu", metrics["dataset_size"])
        col2.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col3.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
        matrix_path = ROOT / "results" / "confusion_matrix.png"
        if matrix_path.exists():
            st.image(str(matrix_path), caption="Ma trận nhầm lẫn trên tập kiểm tra", width=650)
    st.warning(
        "Dữ liệu đi kèm là dữ liệu minh họa cho bài tập lớn. Khi triển khai thực tế, "
        "cần bổ sung dữ liệu đúng lĩnh vực và gán nhãn thủ công."
    )
