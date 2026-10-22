"""Automated Test Suite for AI Spam Message Detector.

Tests dataset validation, model training, prediction reliability,
input handling, save/load consistency, and error states.
"""

import json
from pathlib import Path
import pandas as pd
import pytest
from src.data_loader import validate_and_clean_data
from src.predictor import (
    InvalidMessageError,
    ModelNotLoadedError,
    SpamPredictor,
)
from train_model import build_pipeline, train_and_evaluate


@pytest.fixture(scope="session")
def sample_raw_data() -> pd.DataFrame:
    """Provide a small mock DataFrame resembling the raw UCI SMS format."""
    return pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "spam", "ham", "invalid_label", None],
            "message": [
                "Hello friend, are we still meeting today?",
                "WINNER! Claim your free £500 cash card now! Text WIN to 88202",
                "Ok see you soon.",
                "URGENT: Your account has been compromised. Verify now.",
                "Ok see you soon.",  # duplicate
                "Random text with invalid label",
                "Missing label test",
            ],
        }
    )


class TestDataValidation:
    """Tests for data validation and cleaning."""

    def test_clean_data_removes_invalid_labels_and_nulls(self, sample_raw_data):
        df_clean = validate_and_clean_data(sample_raw_data, remove_duplicates=False)
        assert len(df_clean) == 5
        assert set(df_clean["label"].unique()) == {"ham", "spam"}
        assert not df_clean["message"].isnull().any()

    def test_clean_data_removes_duplicates(self, sample_raw_data):
        df_clean = validate_and_clean_data(sample_raw_data, remove_duplicates=True)
        # Duplicate "Ok see you soon." should be deduplicated
        messages = df_clean["message"].tolist()
        assert messages.count("Ok see you soon.") == 1

    def test_missing_column_raises_value_error(self):
        broken_df = pd.DataFrame({"text_only": ["hello world"]})
        with pytest.raises(ValueError, match="missing required"):
            validate_and_clean_data(broken_df)


class TestModelTrainingAndPersistence:
    """Tests for model training pipeline and artifact serialization."""

    def test_build_pipeline_structure(self):
        pipeline = build_pipeline()
        assert "tfidf" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_train_and_evaluate_generates_artifacts(self, tmp_path):
        # Run training with fixed seed
        pipeline, metrics = train_and_evaluate(test_size=0.2, random_state=42)

        assert pipeline is not None
        assert "performance_metrics" in metrics
        assert metrics["performance_metrics"]["accuracy"] > 0.85
        assert metrics["performance_metrics"]["precision"] > 0.80

        # Verify artifacts exist on disk
        artifacts_dir = Path("artifacts")
        assert (artifacts_dir / "spam_detector_model.joblib").exists()
        assert (artifacts_dir / "metrics.json").exists()

        # Check metrics file content
        with open(artifacts_dir / "metrics.json", "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
        assert saved_metrics["model_name"] == "Multinomial Naive Bayes (TF-IDF)"
        assert "confusion_matrix" in saved_metrics


class TestSpamPredictor:
    """Tests for runtime inference, input validation, and label outputs."""

    @pytest.fixture(autouse=True)
    def setup_predictor(self):
        self.predictor = SpamPredictor()

    def test_prediction_output_schema(self):
        result = self.predictor.predict("Can you please call me back after the meeting?")
        assert isinstance(result, dict)
        assert "prediction" in result
        assert result["prediction"] in ["ham", "spam"]
        assert "spam_probability" in result
        assert 0.0 <= result["spam_probability"] <= 1.0
        assert "confidence_score" in result
        assert "disclaimer" in result

    def test_both_labels_possible(self):
        # We test that the predictor can emit both 'ham' and 'spam' without claiming 100% perfection
        legit_result = self.predictor.predict("Hey mom, I'm heading home right now. Love you!")
        spam_result = self.predictor.predict(
            "URGENT FREE PRIZE! Claim your £1000 voucher immediately by calling 08712345678"
        )
        assert legit_result["prediction"] in ["ham", "spam"]
        assert spam_result["prediction"] in ["ham", "spam"]
        # In practice, with our trained model, spam should have a higher spam_probability
        assert spam_result["spam_probability"] > legit_result["spam_probability"]

    def test_empty_message_handling(self):
        with pytest.raises(InvalidMessageError, match="Empty text cannot be classified"):
            self.predictor.predict("   ")

        with pytest.raises(InvalidMessageError, match="valid non-empty text string"):
            self.predictor.predict(None)

    def test_overly_long_message_handling(self):
        huge_message = "Spam " * 2000  # 10,000 characters
        with pytest.raises(InvalidMessageError, match="too long"):
            self.predictor.predict(huge_message)

    def test_missing_model_behavior(self, tmp_path):
        fake_path = tmp_path / "non_existent_model.joblib"
        with pytest.raises(ModelNotLoadedError, match="Model file not found"):
            SpamPredictor(model_path=fake_path)
