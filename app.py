"""FastAPI Responsive Web Application and REST API.

Provides endpoints for SMS spam prediction, evaluation metrics, and serves
the mobile-first responsive HTML5/CSS3/JavaScript web interface.
"""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predictor import InvalidMessageError, ModelNotLoadedError, get_predictor

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

app = FastAPI(
    title="AI Spam Message Detector API",
    description="Machine Learning REST API and Web Interface for classifying SMS messages as Spam or Ham.",
    version="1.0.0",
)

# CORS configuration - allows cross-origin requests from local frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class MessagePayload(BaseModel):
    """Input schema for spam prediction requests."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The raw SMS message text to evaluate.",
        json_schema_extra={"example": "Congratulations! You have won a £1,000 cash prize. Call now!"},
    )


class PredictionResponse(BaseModel):
    """Output schema for spam prediction responses."""

    message: str
    prediction: str
    is_spam: bool
    spam_probability: float
    ham_probability: float
    confidence_score: float
    explanation: str
    character_count: int
    word_count: int
    analysis_details: dict[str, str] = Field(default_factory=dict)
    disclaimer: str


@app.get("/", summary="Serve the Responsive Web Application")
async def serve_index():
    """Serve the primary responsive single-page web interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html file not found.",
        )
    return FileResponse(str(index_file))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Spam or Ham for an SMS message",
    status_code=status.HTTP_200_OK,
)
async def predict_message(payload: MessagePayload):
    """Accepts an SMS message, evaluates it via the trained pipeline,

    and returns predicted label and probability metrics.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(payload.message)
        return result
    except InvalidMessageError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err
    except ModelNotLoadedError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        ) from err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction.",
        ) from exc


@app.get("/metrics", summary="Get Model Evaluation Metrics")
async def get_metrics():
    """Retrieve actual evaluation metrics saved during model training."""
    if not METRICS_PATH.exists():
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "pending_training",
                "message": "Metrics not generated yet. Run 'python train_model.py' to generate.",
            },
        )
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(status_code=status.HTTP_200_OK, content=data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to read metrics artifact: {exc}",
        ) from exc


@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Check system readiness and model loading status."""
    try:
        predictor = get_predictor()
        model_ready = predictor.is_ready
    except Exception:
        model_ready = False

    return {
        "status": "healthy",
        "model_loaded": model_ready,
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting AI Spam Detector Web Server on http://localhost:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
