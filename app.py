"""FastAPI Responsive Web Application and REST API.

Provides endpoints for SMS spam prediction, batch classification, evaluation metrics,
persistent SQLite history management, and serves the mobile-first responsive HTML5/CSS3/JavaScript interface.
"""

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import database
from src.predictor import InvalidMessageError, ModelNotLoadedError, get_predictor

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database schema is initialized on startup."""
    database.init_db()
    yield


app = FastAPI(
    title="AI Spam Message Detector API",
    description="Machine Learning REST API and Web Interface for classifying SMS messages as Spam or Ham.",
    version="1.2.0",
    lifespan=lifespan,
)

# CORS configuration - allows cross-origin requests from frontends
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


class BatchPayload(BaseModel):
    """Input schema for JSON-based batch spam classification."""

    messages: list[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of raw SMS messages to classify in bulk.",
    )


class FeedbackPayload(BaseModel):
    """Input schema for user feedback / label correction."""

    feedback: str = Field(
        ...,
        pattern="^(correct|misclassified)$",
        description="User assessment: 'correct' or 'misclassified'.",
    )


class PredictionResponse(BaseModel):
    """Output schema for spam prediction responses."""

    id: Optional[int] = None
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

    saves the record to persistent history, and returns predicted metrics.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(payload.message)

        # Persist prediction in real SQLite database
        try:
            record_id = database.save_prediction(result)
            result["id"] = record_id
        except Exception:
            result["id"] = None

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
            detail=f"An unexpected error occurred during prediction: {exc}",
        ) from exc


@app.post(
    "/api/batch-predict",
    summary="Batch classify multiple messages via JSON",
    status_code=status.HTTP_200_OK,
)
async def batch_predict(payload: BatchPayload):
    """Classifies up to 500 SMS messages in a single request and saves results to DB."""
    try:
        predictor = get_predictor()
        results = []
        spam_count = 0
        ham_count = 0

        for msg in payload.messages:
            msg_clean = msg.strip()
            if not msg_clean:
                continue
            res = predictor.predict(msg_clean)
            try:
                rec_id = database.save_prediction(res)
                res["id"] = rec_id
            except Exception:
                res["id"] = None

            if res["is_spam"]:
                spam_count += 1
            else:
                ham_count += 1
            results.append(res)

        total = len(results)
        return {
            "total_processed": total,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "spam_percentage": round((spam_count / total * 100), 1) if total > 0 else 0.0,
            "ham_percentage": round((ham_count / total * 100), 1) if total > 0 else 0.0,
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc


@app.post(
    "/api/batch-upload",
    summary="Batch classify messages by uploading a CSV or TXT file",
    status_code=status.HTTP_200_OK,
)
async def batch_upload_file(file: UploadFile = File(...)):
    """Accepts a .csv or .txt file with messages, extracts, classifies, and saves them."""
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a .csv or .txt file.",
        )

    content = await file.read()
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content.decode("latin-1")

    messages = []
    if file.filename.lower().endswith(".csv"):
        import csv

        reader = csv.reader(io.StringIO(text_content))
        for row in reader:
            if not row:
                continue
            # Look for a message column or take first non-empty cell
            val = row[0].strip()
            # Skip common header rows
            if val.lower() in ["message", "text", "sms", "content", "body"]:
                continue
            if val:
                messages.append(val)
    else:
        # .txt: one message per non-empty line
        for line in text_content.splitlines():
            line_str = line.strip()
            if line_str:
                messages.append(line_str)

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file contains no valid messages.",
        )

    # Limit to 500 messages per upload to ensure fast response
    messages = messages[:500]

    predictor = get_predictor()
    results = []
    spam_count = 0
    ham_count = 0

    for msg in messages:
        try:
            res = predictor.predict(msg)
            try:
                rec_id = database.save_prediction(res)
                res["id"] = rec_id
            except Exception:
                res["id"] = None

            if res["is_spam"]:
                spam_count += 1
            else:
                ham_count += 1
            results.append(res)
        except Exception:
            continue

    total = len(results)
    return {
        "filename": file.filename,
        "total_processed": total,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_percentage": round((spam_count / total * 100), 1) if total > 0 else 0.0,
        "ham_percentage": round((ham_count / total * 100), 1) if total > 0 else 0.0,
        "results": results,
    }


@app.get("/api/history", summary="Get Paginated Prediction History")
async def get_history_records(
    search: Optional[str] = Query(None, description="Search term in messages"),
    filter_type: str = Query("all", pattern="^(all|spam|ham)$", description="Filter by spam/ham/all"),
    sort: str = Query(
        "newest",
        pattern="^(newest|oldest|highest_confidence|lowest_confidence)$",
        description="Sort order",
    ),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """Retrieve history records with search, filtering, and sorting."""
    try:
        return database.get_history(
            search=search,
            filter_type=filter_type,
            sort=sort,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history: {exc}",
        ) from exc


@app.get("/api/history/stats", summary="Get Real Database Statistics")
async def get_history_statistics():
    """Retrieve actual cumulative counts and percentages from real stored records."""
    try:
        return database.get_statistics()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch statistics: {exc}",
        ) from exc


@app.delete("/api/history/{record_id}", summary="Delete a Single History Record")
async def delete_history_record(record_id: int):
    """Delete a single prediction record from the database."""
    deleted = database.delete_prediction(record_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Record not found or already deleted.",
        )
    return {"status": "success", "message": f"Record #{record_id} deleted successfully."}


@app.delete("/api/history", summary="Clear All History Records")
async def clear_all_history():
    """Remove all records from prediction history."""
    deleted_count = database.clear_history()
    return {
        "status": "success",
        "deleted_count": deleted_count,
        "message": f"Successfully cleared {deleted_count} records from history.",
    }


@app.post("/api/history/{record_id}/feedback", summary="Submit Prediction Feedback")
async def submit_feedback(record_id: int, payload: FeedbackPayload):
    """Record whether user verified prediction was correct or misclassified."""
    updated = database.update_feedback(record_id, payload.feedback)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Record not found.",
        )
    return {"status": "success", "message": "Feedback recorded. Thank you!"}


@app.get("/api/history/export", summary="Export Prediction History as CSV")
async def export_history():
    """Download the full prediction history as a CSV file."""
    try:
        csv_data = database.export_history_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=spam_detector_history.csv"},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV export failed: {exc}",
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
    """Check system readiness, model loading status, and database connection."""
    try:
        predictor = get_predictor()
        model_ready = predictor.is_ready
    except Exception:
        model_ready = False

    try:
        db_stats = database.get_statistics()
        db_ready = True
    except Exception:
        db_ready = False
        db_stats = {}

    return {
        "status": "healthy" if (model_ready and db_ready) else "degraded",
        "model_loaded": model_ready,
        "database_connected": db_ready,
        "records_count": db_stats.get("total_checked", 0),
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting AI Spam Message Detector on http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
