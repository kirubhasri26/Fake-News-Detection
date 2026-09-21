"""
train_model.py
----------------
This script trains a simple Fake News Detection model using:
    TF-IDF Vectorizer + Logistic Regression

Steps performed:
    1. Load the dataset (dataset/news.csv)
    2. Validate required columns
    3. Clean missing values
    4. Combine title + text into a single field
    5. Split into training/testing sets
    6. Convert text into TF-IDF features
    7. Train a Logistic Regression classifier
    8. Evaluate accuracy on the test set
    9. Save the trained model  -> model/model.pkl
   10. Save the fitted vectorizer -> model/vectorizer.pkl

IMPORTANT NOTE ABOUT THE DATASET:
    dataset/news.csv included with this project is a SMALL DEMO DATASET
    created only so the project can be trained, run, and demonstrated
    end-to-end. It is NOT a real-world, production-grade fake news
    dataset, and the accuracy reported below should NOT be interpreted
    as representative of real-world performance. For a production
    system, replace dataset/news.csv with a large, well-labeled,
    real-world dataset (for example, a public Kaggle "Fake and Real
    News" dataset) that has the same column names: title, text, label.
"""

import os
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

REQUIRED_COLUMNS = {"title", "text", "label"}


def clean_text(text: str) -> str:
    """Lowercase text and remove characters that are not useful for TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # keep only letters/numbers
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


def main():
    print("Training started...")

    # -----------------------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------------------
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at: {DATASET_PATH}")
        print("Please make sure dataset/news.csv exists before training.")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded... ({len(df)} rows)")

    # -----------------------------------------------------------------
    # 2. Validate required columns
    # -----------------------------------------------------------------
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"ERROR: Dataset is missing required columns: {missing_cols}")
        print("The dataset must contain the columns: title, text, label")
        sys.exit(1)

    # -----------------------------------------------------------------
    # 3. Clean missing values
    # -----------------------------------------------------------------
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(["REAL", "FAKE"])]

    if df.empty:
        print("ERROR: No valid rows remain after cleaning the dataset.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # 4. Combine title + text, then clean
    # -----------------------------------------------------------------
    df["combined_text"] = (df["title"] + " " + df["text"]).apply(clean_text)

    X = df["combined_text"]
    y = df["label"]

    # -----------------------------------------------------------------
    # 5. Split into train/test sets
    # -----------------------------------------------------------------
    test_size = 0.2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback for very small datasets that can't be stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # -----------------------------------------------------------------
    # 6. TF-IDF vectorization
    # -----------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -----------------------------------------------------------------
    # 7. Train Logistic Regression
    # -----------------------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # -----------------------------------------------------------------
    # 8. Evaluate
    # -----------------------------------------------------------------
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    print("Training completed...")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(
        "NOTE: This accuracy is measured on a small DEMO dataset and is "
        "for demonstration purposes only. It does not represent "
        "real-world fake news detection performance."
    )

    # -----------------------------------------------------------------
    # 9 & 10. Save model and vectorizer
    # -----------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Model saved successfully -> {MODEL_PATH}")
    print(f"Vectorizer saved successfully -> {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
