"""Inference and Prediction Service for Spam Classification.

Loads the pre-trained Scikit-learn Pipeline and executes predictions
with probability estimation, input validation, and safety boundaries.
"""

from pathlib import Path
from typing import Any
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "spam_detector_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

MAX_MESSAGE_LENGTH = 5000  # Reasonable upper boundary for SMS / messaging text


class ModelNotLoadedError(Exception):
    """Raised when the trained model artifact is missing or unreadable."""


class InvalidMessageError(ValueError):
    """Raised when user input violates validation constraints."""


class SpamPredictor:
    """Encapsulates model loading, input validation, and inference."""

    _instance = None
    _pipeline = None

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or MODEL_PATH
        self.load_model()

    def load_model(self) -> None:
        """Load the persisted joblib model pipeline."""
        if not self.model_path.exists():
            raise ModelNotLoadedError(
                f"Model file not found at '{self.model_path}'. "
                "Please train the model first by running: python train_model.py"
            )
        try:
            self._pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise ModelNotLoadedError(f"Failed to load model artifact: {exc}") from exc

    @property
    def is_ready(self) -> bool:
        """Check if model pipeline is loaded and ready for inference."""
        return self._pipeline is not None

    def validate_message(self, message: Any) -> str:
        """Validate input message string.

        Args:
            message: Input text to validate.

        Returns:
            Sanitized, stripped message string.

        Raises:
            InvalidMessageError: On empty, wrong type, or overly long inputs.
        """
        if message is None or not isinstance(message, str):
            raise InvalidMessageError("The message must be a valid non-empty text string.")

        sanitized = message.strip()
        if not sanitized:
            raise InvalidMessageError("Please enter a message. Empty text cannot be classified.")

        if len(sanitized) > MAX_MESSAGE_LENGTH:
            raise InvalidMessageError(
                f"Message is too long ({len(sanitized)} characters). "
                f"Maximum allowed length is {MAX_MESSAGE_LENGTH} characters."
            )

        return sanitized

    def predict(self, message: str) -> dict[str, Any]:
        """Perform spam classification on a message.

        Args:
            message: Raw text message string.

        Returns:
            Dict containing classification label, probabilities, and metadata.
        """
        clean_text = self.validate_message(message)

        if not self.is_ready:
            self.load_model()

        # Run pipeline inference
        predicted_label = str(self._pipeline.predict([clean_text])[0])
        probabilities = self._pipeline.predict_proba([clean_text])[0]
        classes = list(self._pipeline.classes_)

        # Extract probability for each class safely
        prob_dict = dict(zip(classes, probabilities))
        spam_prob = float(prob_dict.get("spam", 0.0))
        ham_prob = float(prob_dict.get("ham", 0.0))

        is_spam = predicted_label == "spam"
        confidence = spam_prob if is_spam else ham_prob

        # Contextual explanation based on probability thresholds
        if is_spam:
            if spam_prob >= 0.85:
                explanation = "High confidence spam patterns detected (e.g., promotional terms, urgency, or links)."
            else:
                explanation = "Moderate spam indicators found. Exercise caution before opening links or replying."
        else:
            if ham_prob >= 0.85:
                explanation = "Standard conversational or legitimate communication patterns detected."
            else:
                explanation = "Appears legitimate, but borderline wording was detected."

        return {
            "message": clean_text,
            "prediction": predicted_label,
            "is_spam": is_spam,
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "confidence_score": round(confidence * 100, 2),
            "explanation": explanation,
            "character_count": len(clean_text),
            "word_count": len(clean_text.split()),
            "disclaimer": (
                "Notice: Model predictions are statistical estimates based on historical SMS data "
                "and are not an absolute guarantee of safety or malicious intent."
            ),
        }


# Singleton instance helper for efficient reuse across requests
_predictor_instance: SpamPredictor | None = None


def get_predictor() -> SpamPredictor:
    """Return the shared SpamPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SpamPredictor()
    return _predictor_instance
