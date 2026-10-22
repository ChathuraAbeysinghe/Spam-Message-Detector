"""Model Training, Evaluation, and Serialization Pipeline.

Trains a Scikit-learn Pipeline consisting of TF-IDF text vectorization
and a Multinomial Naive Bayes classifier on the UCI SMS Spam Collection.
Evaluates model performance using stratified splitting and exports the
trained pipeline and performance metrics into the artifacts directory.
"""

from datetime import datetime
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data_loader import get_dataset

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "spam_detector_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

RANDOM_SEED = 42


def build_pipeline() -> Pipeline:
    """Create a reproducible Scikit-learn pipeline.

    Combines TF-IDF vectorization and Multinomial Naive Bayes classifier.
    Fitting occurs strictly on training data to prevent data leakage.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    ngram_range=(1, 2),  # unigrams and bigrams
                    max_features=5000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )


def train_and_evaluate(
    test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> tuple[Pipeline, dict]:
    """Train the model, evaluate against unseen test data, and return results.

    Args:
        test_size: Fraction of data reserved for testing (default: 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        tuple of (fitted_pipeline, metrics_dictionary)
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading clean dataset...", flush=True)
    df = get_dataset()

    X = df["message"]
    y = df["label"]

    print(
        f"\nStep 2: Performing stratified train/test split ({100*(1-test_size):.0f}% train, {100*test_size:.0f}% test)...",
        flush=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}", flush=True)
    print(f"Train distribution:\n{y_train.value_counts(normalize=True).round(3)}", flush=True)

    print("\nStep 3: Training Primary Model (TF-IDF + Multinomial Naive Bayes)...", flush=True)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Step 4: Evaluating on unseen test set...", flush=True)
    y_pred = pipeline.predict(X_test)

    # Compute key evaluation metrics for 'spam' as positive class
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, pos_label="spam"))
    recall = float(recall_score(y_test, y_pred, pos_label="spam"))
    f1 = float(f1_score(y_test, y_pred, pos_label="spam"))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report = classification_report(y_test, y_pred, output_dict=True)

    # Optional model comparison with Logistic Regression
    print("\nStep 5: Training Benchmark Classifier (TF-IDF + Logistic Regression)...")
    benchmark_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(random_state=random_state)),
        ]
    )
    benchmark_pipeline.fit(X_train, y_train)
    benchmark_pred = benchmark_pipeline.predict(X_test)
    benchmark_acc = float(accuracy_score(y_test, benchmark_pred))
    benchmark_f1 = float(f1_score(y_test, benchmark_pred, pos_label="spam"))

    # Compile structured metrics dictionary
    metrics = {
        "model_name": "Multinomial Naive Bayes (TF-IDF)",
        "random_seed": random_state,
        "evaluation_timestamp": datetime.now().isoformat(),
        "dataset_summary": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "ham_count": int((df["label"] == "ham").sum()),
            "spam_count": int((df["label"] == "spam").sum()),
        },
        "performance_metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "confusion_matrix": {
            "labels": ["ham", "spam"],
            "matrix": [[tn, fp], [fn, tp]],
            "true_negatives": tn,  # Legit messages correctly identified
            "false_positives": fp,  # Legit messages wrongly flagged as spam
            "false_negatives": fn,  # Spam messages missed
            "true_positives": tp,  # Spam messages correctly caught
        },
        "error_analysis_explanation": {
            "false_positive": (
                "A legitimate message (Ham) is mistakenly predicted as Spam. "
                "In real life, this is critical because a user might miss an urgent "
                "bank OTP, job offer, or family message. We prioritize High Precision."
            ),
            "false_negative": (
                "A Spam message slips through and is predicted as Ham. "
                "While annoying, the user simply sees an unwanted message. "
                "Less damaging than losing an essential message."
            ),
        },
        "benchmark_comparison": {
            "benchmark_model": "Logistic Regression (TF-IDF)",
            "accuracy": round(benchmark_acc, 4),
            "f1_score": round(benchmark_f1, 4),
        },
        "classification_report": report,
    }

    print("\n--- Model Evaluation Summary ---", flush=True)
    print(f"Accuracy:  {accuracy * 100:.2f}%", flush=True)
    print(f"Precision: {precision * 100:.2f}% (Spam)", flush=True)
    print(f"Recall:    {recall * 100:.2f}% (Spam)", flush=True)
    print(f"F1-Score:  {f1 * 100:.2f}% (Spam)", flush=True)
    print(f"Confusion Matrix [ [TN={tn}, FP={fp}], [FN={fn}, TP={tp}] ]", flush=True)
    print(f"Benchmark Logistic Regression Accuracy: {benchmark_acc * 100:.2f}%", flush=True)

    # Serialize trained pipeline and metrics
    print(f"\nStep 6: Saving model to {MODEL_PATH}...", flush=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Saving metrics report to {METRICS_PATH}...", flush=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Pipeline training and artifact generation complete!", flush=True)
    return pipeline, metrics


if __name__ == "__main__":
    train_and_evaluate()
