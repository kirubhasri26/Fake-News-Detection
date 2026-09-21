"""
database.py
------------
Small helper module that manages the SQLite database used by the
Fake News Detection application.

Tables:
    predictions  - stores every news analysis performed by a user
    feedback     - stores messages submitted through the Contact page

The database file is created automatically (if it does not already
exist) the first time the Flask application starts.
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "database.db")


def get_connection():
    """Return a new SQLite connection with row access by column name."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the required tables if they do not already exist."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def save_prediction(title, content, prediction, confidence):
    """Insert a new prediction record and return its new row id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (title, content, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (title, content, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_all_predictions(limit=200):
    """Return the most recent prediction records, newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, prediction, confidence, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def save_feedback(name, email, message):
    """Insert a new feedback/contact record and return its new row id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO feedback (name, email, message, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (name, email, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id
