"""Integration and API tests for FastAPI endpoints in app.py."""

from fastapi.testclient import TestClient
import pytest
from app import app
from src import database


@pytest.fixture
def client():
    """Create test client for FastAPI."""
    return TestClient(app)


def test_health_check_endpoint(client):
    """Verify health endpoint reports correct structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "database_connected" in data


def test_predict_endpoint_valid_ham(client):
    """Verify predict returns valid payload and status 200 for ham."""
    payload = {"message": "Hey friend, are we having lunch at noon?"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["ham", "spam"]
    assert "is_spam" in data
    assert "confidence_score" in data
    assert "analysis_details" in data
    assert "id" in data


def test_predict_endpoint_valid_spam(client):
    """Verify predict returns valid payload for high-confidence spam."""
    payload = {"message": "WINNER! You have won £1,000 cash. Call 09061701461 to claim your prize."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "spam"
    assert data["is_spam"] is True
    assert data["spam_probability"] > 0.5


def test_predict_endpoint_empty_message(client):
    """Verify 400 or 422 error on empty message."""
    response = client.post("/predict", json={"message": ""})
    assert response.status_code in [400, 422]


def test_history_crud_flow(client):
    """Verify full history lifecycle: predict -> list -> get stats -> delete -> clear."""
    # 1. Make a prediction
    res = client.post("/predict", json={"message": "Temporary integration test SMS message."})
    assert res.status_code == 200
    rec_id = res.json().get("id")

    # 2. Get history list
    hist_res = client.get("/api/history?limit=10")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert "records" in hist_data
    assert hist_data["total"] >= 1

    # 3. Get history statistics
    stats_res = client.get("/api/history/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_checked"] >= 1

    # 4. Submit feedback
    if rec_id:
        fb_res = client.post(f"/api/history/{rec_id}/feedback", json={"feedback": "correct"})
        assert fb_res.status_code == 200

    # 5. Delete specific record
    if rec_id:
        del_res = client.delete(f"/api/history/{rec_id}")
        assert del_res.status_code == 200

    # 6. Export CSV
    export_res = client.get("/api/history/export")
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers.get("content-type", "")


def test_batch_predict_json(client):
    """Verify batch prediction with multiple messages."""
    payload = {
        "messages": [
            "Are you coming to class today?",
            "URGENT: Win $500 Walmart gift card! Click here now!",
            "Can you pick up groceries?",
        ]
    }
    response = client.post("/api/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    assert len(data["results"]) == 3
    assert "spam_count" in data
    assert "ham_count" in data


def test_metrics_endpoint(client):
    """Verify /metrics returns performance data."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    if data.get("status") != "pending_training":
        assert "performance_metrics" in data
