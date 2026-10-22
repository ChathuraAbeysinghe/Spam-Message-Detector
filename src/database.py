"""SQLite Database Layer for AI Spam Message Detector.

Provides persistent storage, retrieval, filtering, pagination, statistics,
and CSV export for SMS spam classification history.
"""

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "spam_detector.db"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create and return a SQLite database connection with row factory."""
    target_path = db_path or DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and indexes."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    is_spam INTEGER NOT NULL,
                    spam_probability REAL NOT NULL,
                    ham_probability REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    suspicious_keywords TEXT,
                    links_detected TEXT,
                    sender_analysis TEXT,
                    message_pattern TEXT,
                    character_count INTEGER,
                    word_count INTEGER,
                    user_feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_created_at
                ON prediction_history (created_at DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_is_spam
                ON prediction_history (is_spam);
                """
            )
    finally:
        conn.close()


def save_prediction(result: dict[str, Any], db_path: Optional[Path] = None) -> int:
    """Save an inference prediction result to the database."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        details = result.get("analysis_details", {}) or {}
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO prediction_history (
                    message, prediction, is_spam, spam_probability,
                    ham_probability, confidence_score, suspicious_keywords,
                    links_detected, sender_analysis, message_pattern,
                    character_count, word_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["message"],
                    result["prediction"],
                    1 if result["is_spam"] else 0,
                    float(result["spam_probability"]),
                    float(result["ham_probability"]),
                    float(result["confidence_score"]),
                    details.get("suspicious_keywords", "None found"),
                    details.get("links_detected", "None found"),
                    details.get("sender_analysis", "Looks legitimate"),
                    details.get("message_pattern", "Normal conversation pattern"),
                    int(result.get("character_count", len(result["message"]))),
                    int(result.get("word_count", len(result["message"].split()))),
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_history(
    search: Optional[str] = None,
    filter_type: str = "all",
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Retrieve filtered, searched, and paginated prediction history."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        query_conditions = []
        params: list[Any] = []

        if search and search.strip():
            query_conditions.append("message LIKE ?")
            params.append(f"%{search.strip()}%")

        if filter_type == "spam":
            query_conditions.append("is_spam = 1")
        elif filter_type == "ham":
            query_conditions.append("is_spam = 0")

        where_clause = ""
        if query_conditions:
            where_clause = "WHERE " + " AND ".join(query_conditions)

        # Count total matching rows
        count_query = f"SELECT COUNT(*) AS total FROM prediction_history {where_clause}"
        cursor = conn.execute(count_query, params)
        total_records = cursor.fetchone()["total"]

        # Determine order
        order_by = "created_at DESC"
        if sort == "oldest":
            order_by = "created_at ASC"
        elif sort == "highest_confidence":
            order_by = "confidence_score DESC"
        elif sort == "lowest_confidence":
            order_by = "confidence_score ASC"

        data_query = f"""
            SELECT * FROM prediction_history
            {where_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
        """
        cursor = conn.execute(data_query, params + [limit, offset])
        rows = [dict(row) for row in cursor.fetchall()]

        # Format records for JSON response
        formatted_rows = []
        for r in rows:
            formatted_rows.append(
                {
                    "id": r["id"],
                    "message": r["message"],
                    "prediction": r["prediction"],
                    "is_spam": bool(r["is_spam"]),
                    "spam_probability": r["spam_probability"],
                    "ham_probability": r["ham_probability"],
                    "confidence_score": r["confidence_score"],
                    "character_count": r["character_count"],
                    "word_count": r["word_count"],
                    "analysis_details": {
                        "suspicious_keywords": r["suspicious_keywords"],
                        "links_detected": r["links_detected"],
                        "sender_analysis": r["sender_analysis"],
                        "message_pattern": r["message_pattern"],
                    },
                    "user_feedback": r["user_feedback"],
                    "created_at": r["created_at"],
                }
            )

        return {
            "total": total_records,
            "limit": limit,
            "offset": offset,
            "records": formatted_rows,
        }
    finally:
        conn.close()


def get_statistics(db_path: Optional[Path] = None) -> dict[str, Any]:
    """Calculate actual aggregate statistics from real stored records."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT 
                COUNT(*) AS total_checked,
                COALESCE(SUM(CASE WHEN is_spam = 1 THEN 1 ELSE 0 END), 0) AS spam_count,
                COALESCE(SUM(CASE WHEN is_spam = 0 THEN 1 ELSE 0 END), 0) AS ham_count
            FROM prediction_history
            """
        )
        row = cursor.fetchone()
        total = row["total_checked"]
        spam = row["spam_count"]
        ham = row["ham_count"]

        if total > 0:
            spam_pct = round((spam / total) * 100, 1)
            ham_pct = round((ham / total) * 100, 1)
        else:
            spam_pct = 0.0
            ham_pct = 0.0

        return {
            "total_checked": total,
            "spam_count": spam,
            "ham_count": ham,
            "spam_percentage": spam_pct,
            "ham_percentage": ham_pct,
        }
    finally:
        conn.close()


def delete_prediction(record_id: int, db_path: Optional[Path] = None) -> bool:
    """Delete a single prediction record by ID."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_history(db_path: Optional[Path] = None) -> int:
    """Clear all records from prediction history."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM prediction_history")
            return cursor.rowcount
    finally:
        conn.close()


def update_feedback(record_id: int, feedback: str, db_path: Optional[Path] = None) -> bool:
    """Record user feedback ('correct' or 'misclassified') for a prediction."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "UPDATE prediction_history SET user_feedback = ? WHERE id = ?",
                (feedback, record_id),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def export_history_csv(db_path: Optional[Path] = None) -> str:
    """Generate a CSV string containing all stored prediction records."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, message, prediction, is_spam, spam_probability,
                   ham_probability, confidence_score, suspicious_keywords,
                   links_detected, sender_analysis, message_pattern,
                   character_count, word_count, user_feedback, created_at
            FROM prediction_history
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "ID",
                "Message",
                "Prediction",
                "Is Spam",
                "Spam Probability (%)",
                "Ham Probability (%)",
                "Confidence Score (%)",
                "Suspicious Keywords",
                "Links Detected",
                "Sender Analysis",
                "Message Pattern",
                "Character Count",
                "Word Count",
                "User Feedback",
                "Created At (UTC)",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["id"],
                    r["message"],
                    r["prediction"],
                    "Yes" if r["is_spam"] else "No",
                    f"{r['spam_probability'] * 100:.2f}",
                    f"{r['ham_probability'] * 100:.2f}",
                    f"{r['confidence_score']:.2f}",
                    r["suspicious_keywords"],
                    r["links_detected"],
                    r["sender_analysis"],
                    r["message_pattern"],
                    r["character_count"],
                    r["word_count"],
                    r["user_feedback"] or "None",
                    r["created_at"],
                ]
            )
        return output.getvalue()
    finally:
        conn.close()
