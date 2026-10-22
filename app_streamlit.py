"""Streamlit Web Application for AI Spam Message Detector.

Alternative interactive dashboard for university demonstration,
data exploration, and live classification.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.predictor import InvalidMessageError, ModelNotLoadedError, SpamPredictor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "spam_detector_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

st.set_page_config(
    page_title="AI Spam Message Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor():
    """Cache model predictor instance in Streamlit resource pool."""
    return SpamPredictor(MODEL_PATH)


def load_metrics():
    """Load saved evaluation metrics from JSON artifact."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    # Custom CSS matching reference UI palette
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8fafc;
        }
        h1, h2, h3 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
        }
        .main-header-title {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 800;
        }
        .purple-accent {
            color: #4f46e5;
        }
        .stButton>button {
            background-color: #4f46e5 !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 0.6rem 1.25rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
        }
        .analysis-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header-title">🛡️ Spam <span class="purple-accent">Message Detector</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Enter a message below and let AI check if it's spam or not. "
        "We analyze the content, links, and patterns to keep you safe from scams."
    )

    # Sidebar: Model and Dataset Information
    with st.sidebar:
        st.header("📌 Project Details")
        st.markdown(
            """
            - **Model**: Multinomial Naive Bayes
            - **Feature Extraction**: TF-IDF (1-2 ngrams)
            - **Dataset**: UCI SMS Spam Collection
            - **Evaluation Split**: 80% Train / 20% Stratified Test
            - **Framework**: Scikit-learn + Python 3
            """
        )

        st.subheader("💡 Error Analysis Context")
        st.info(
            "**False Positive (FP)**: Important message mistaken as Spam. Highly disruptive.\n\n"
            "**False Negative (FN)**: Spam slips into inbox. Less disruptive.\n\n"
            "We prioritize **High Precision** to minimize False Positives."
        )

    # Main columns: Left for Prediction, Right for Metrics
    col_pred, col_metrics = st.columns([1.2, 0.8], gap="large")

    with col_pred:
        st.subheader("✉️ Message Classifier")

        # Quick sample selector
        sample_options = {
            "Select a sample message...": "",
            "🚨 Scam: £1,000 Gift Card": (
                "URGENT! You have won a £1,000 Walmart gift card! Call 09061701461 to claim now."
            ),
            "🎁 Promo: Free Subscription": (
                "FreeMsg: Congrats! You are selected for a free trial. Reply YES to 80488. T&C apply."
            ),
            "💬 Ham: Library Meeting": (
                "Hey! Are you still coming to the library for study group this evening? Let me know."
            ),
            "🔑 Ham: Bank OTP": (
                "Your verification code is 492018. It is valid for 10 minutes. Do not share this code."
            ),
        }

        selected_sample = st.selectbox("Quick Test Samples:", list(sample_options.keys()))
        default_text = sample_options[selected_sample] if selected_sample else ""

        message_input = st.text_area(
            "Enter SMS Message Text:",
            value=default_text,
            height=130,
            placeholder="Type or paste an SMS message to classify...",
            max_chars=5000,
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            predict_clicked = st.button("🔍 Classify Message", type="primary", use_container_width=True)
        with col_btn2:
            clear_clicked = st.button("Clear Input", use_container_width=True)

        if clear_clicked:
            st.rerun()

        if predict_clicked:
            if not message_input.strip():
                st.warning("⚠️ Please enter a non-empty SMS message to classify.")
            else:
                try:
                    predictor = load_predictor()
                    result = predictor.predict(message_input)

                    st.markdown("---")
                    if result["is_spam"]:
                        st.error(f"### 🚨 Prediction: SPAM MESSAGE\n**Confidence:** {result['confidence_score']}%")
                    else:
                        st.success(f"### ✅ Prediction: NOT SPAM (HAM)\n**Confidence:** {result['confidence_score']}%")

                    # Probability Progress Bar
                    spam_prob = result["spam_probability"]
                    st.write(f"**Model-Estimated Spam Probability:** {spam_prob * 100:.1f}%")
                    st.progress(spam_prob)

                    st.info(f"**Explanation:** {result['explanation']}")
                    st.caption(f"ℹ️ {result['disclaimer']}")

                except ModelNotLoadedError as err:
                    st.error(f"❌ {err}")
                except InvalidMessageError as err:
                    st.warning(f"⚠️ {err}")
                except Exception as err:
                    st.error(f"❌ Prediction Error: {err}")

    with col_metrics:
        st.subheader("📊 Model Performance")

        metrics_data = load_metrics()

        if metrics_data:
            perf = metrics_data["performance_metrics"]
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{perf['accuracy'] * 100:.2f}%")
            m2.metric("Precision (Spam)", f"{perf['precision'] * 100:.2f}%")

            m3, m4 = st.columns(2)
            m3.metric("Recall (Spam)", f"{perf['recall'] * 100:.2f}%")
            m4.metric("F1-Score (Spam)", f"{perf['f1_score'] * 100:.2f}%")

            # Confusion Matrix Table
            st.markdown("#### 🎯 Test Confusion Matrix")
            cm = metrics_data["confusion_matrix"]
            cm_df = pd.DataFrame(
                [
                    [cm["true_negatives"], cm["false_positives"]],
                    [cm["false_negatives"], cm["true_positives"]],
                ],
                index=["Actual Ham", "Actual Spam"],
                columns=["Pred Ham", "Pred Spam"],
            )
            st.dataframe(cm_df, use_container_width=True)

            if "benchmark_comparison" in metrics_data:
                bench = metrics_data["benchmark_comparison"]
                st.caption(
                    f"**Benchmark Comparison:** {bench['benchmark_model']} "
                    f"achieved {bench['accuracy'] * 100:.2f}% Accuracy."
                )
        else:
            st.warning("Model metrics file not found. Run `python train_model.py` to evaluate.")


if __name__ == "__main__":
    main()
