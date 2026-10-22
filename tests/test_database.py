"""Unit tests for SQLite database layer in src/database.py."""

import csv
import io
from pathlib import Path
import pytest
from src import database


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide a temporary database path for test isolation."""
    db_file = tmp_path / "test_spam.db"
    database.init_db(db_file)
    return db_file


def test_init_db_creates_table(temp_db):
    """Verify tables and indexes are created properly."""
    conn = database.get_db_connection(temp_db)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_history'")
    assert cursor.fetchone() is not None
    conn.close()


def test_save_prediction_and_get_statistics(temp_db):
    """Verify inserting predictions updates statistics accurately."""
    stats_empty = database.get_statistics(temp_db)
    assert stats_empty["total_checked"] == 0
    assert stats_empty["spam_count"] == 0
    assert stats_empty["ham_count"] == 0

    # Insert one spam
    spam_sample = {
        "message": "Win a cash prize now!",
        "prediction": "spam",
        "is_spam": True,
        "spam_probability": 0.98,
        "ham_probability": 0.02,
        "confidence_score": 98.0,
        "analysis_details": {"suspicious_keywords": "prize, cash", "links_detected": "None found"},
        "character_count": 22,
        "word_count": 5,
    }
    rec_id1 = database.save_prediction(spam_sample, temp_db)
    assert rec_id1 > 0

    # Insert one ham
    ham_sample = {
        "message": "See you at the conference tomorrow.",
        "prediction": "ham",
        "is_spam": False,
        "spam_probability": 0.05,
        "ham_probability": 0.95,
        "confidence_score": 95.0,
        "analysis_details": {"suspicious_keywords": "None found", "links_detected": "None found"},
        "character_count": 35,
        "word_count": 6,
    }
    rec_id2 = database.save_prediction(ham_sample, temp_db)
    assert rec_id2 > rec_id1

    stats = database.get_statistics(temp_db)
    assert stats["total_checked"] == 2
    assert stats["spam_count"] == 1
    assert stats["ham_count"] == 1
    assert stats["spam_percentage"] == 50.0
    assert stats["ham_percentage"] == 50.0


def test_get_history_filtering_and_search(temp_db):
    """Verify search, filter, and pagination logic."""
    for i in range(5):
        database.save_prediction(
            {
                "message": f"Special promo offer {i} click here",
                "prediction": "spam",
                "is_spam": True,
                "spam_probability": 0.95,
                "ham_probability": 0.05,
                "confidence_score": 95.0,
                "analysis_details": {},
            },
            temp_db,
        )
    for i in range(3):
        database.save_prediction(
            {
                "message": f"Hey friend {i} how are you",
                "prediction": "ham",
                "is_spam": False,
                "spam_probability": 0.02,
                "ham_probability": 0.98,
                "confidence_score": 98.0,
                "analysis_details": {},
            },
            temp_db,
        )

    # Search
    search_res = database.get_history(search="promo offer", db_path=temp_db)
    assert search_res["total"] == 5

    # Filter spam only
    spam_res = database.get_history(filter_type="spam", db_path=temp_db)
    assert spam_res["total"] == 5

    # Filter ham only
    ham_res = database.get_history(filter_type="ham", db_path=temp_db)
    assert ham_res["total"] == 3

    # Pagination
    page1 = database.get_history(limit=4, offset=0, db_path=temp_db)
    assert len(page1["records"]) == 4
    page2 = database.get_history(limit=4, offset=4, db_path=temp_db)
    assert len(page2["records"]) == 4


def test_delete_and_clear_history(temp_db):
    """Verify single deletion and clearing all history."""
    rec_id = database.save_prediction(
        {
            "message": "Temporary test message",
            "prediction": "ham",
            "is_spam": False,
            "spam_probability": 0.01,
            "ham_probability": 0.99,
            "confidence_score": 99.0,
            "analysis_details": {},
        },
        temp_db,
    )
    assert database.delete_prediction(rec_id, temp_db) is True
    assert database.delete_prediction(9999, temp_db) is False

    # Insert 2 new
    database.save_prediction(
        {
            "message": "Msg 1",
            "prediction": "ham",
            "is_spam": False,
            "spam_probability": 0.01,
            "ham_probability": 0.99,
            "confidence_score": 99.0,
            "analysis_details": {},
        },
        temp_db,
    )
    database.save_prediction(
        {
            "message": "Msg 2",
            "prediction": "spam",
            "is_spam": True,
            "spam_probability": 0.9,
            "ham_probability": 0.1,
            "confidence_score": 90.0,
            "analysis_details": {},
        },
        temp_db,
    )

    deleted_count = database.clear_history(temp_db)
    assert deleted_count == 2
    assert database.get_statistics(temp_db)["total_checked"] == 0


def test_export_history_csv(temp_db):
    """Verify CSV export contains correct headers and rows."""
    database.save_prediction(
        {
            "message": "Win a lottery",
            "prediction": "spam",
            "is_spam": True,
            "spam_probability": 0.95,
            "ham_probability": 0.05,
            "confidence_score": 95.0,
            "analysis_details": {"suspicious_keywords": "lottery"},
        },
        temp_db,
    )
    csv_str = database.export_history_csv(temp_db)
    reader = csv.reader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 2  # header + 1 record
    assert "Message" in rows[0]
    assert "Prediction" in rows[0]
    assert rows[1][1] == "Win a lottery"
    assert rows[1][2] == "spam"
