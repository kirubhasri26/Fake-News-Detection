"""
app.py
------
Main Flask application for the AI-Based Fake News Detection System.

Routes:
    /               - Home page
    /predict        - News analysis form (GET) and result (POST)
    /history        - Prediction history page
    /about          - About page
    /contact        - Contact / feedback page

The app loads a pre-trained TF-IDF vectorizer and Logistic Regression
model from the model/ folder. If those files do not exist yet, the
app still starts but shows a friendly message asking the user to run
the training script first.
"""

import os
import re
import traceback
from html import escape

import joblib
from flask import Flask, flash, redirect, render_template, request, url_for

from database.database import (
    get_all_predictions,
    init_db,
    save_feedback,
    save_prediction,
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

MAX_CONTENT_CHARACTERS = 20000   # simple protection against extremely large input
MIN_CONTENT_CHARACTERS = 20      # minimum content length for a meaningful prediction

app = Flask(__name__)
# SECRET_KEY is read from the environment; a random fallback is used for
# local development only. Always set FLASK_SECRET_KEY in production.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# ---------------------------------------------------------------------------
# Load ML model + vectorizer (if available)
# ---------------------------------------------------------------------------
model = None
vectorizer = None
MODEL_LOAD_ERROR = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        MODEL_LOAD_ERROR = (
            "The machine-learning model files were not found. "
            "Please train the model first by running: "
            "python training/train_model.py"
        )
except Exception:
    MODEL_LOAD_ERROR = (
        "The machine-learning model could not be loaded. "
        "Please retrain the model by running: python training/train_model.py"
    )
    traceback.print_exc()

# Ensure the SQLite database and tables exist before the app serves requests.
try:
    init_db()
except Exception:
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Apply the same text-cleaning logic used during training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_news(title: str, content: str):
    """
    Run the trained model on the combined title + content.
    Returns a tuple: (prediction_label, confidence_percentage)
    """
    combined = clean_text(f"{title} {content}")
    features = vectorizer.transform([combined])

    prediction = model.predict(features)[0]

    # Try to get a genuine probability/confidence score from the model.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100
    elif hasattr(model, "decision_function"):
        # Fallback: convert the decision function score into a 0-100 range
        score = model.decision_function(features)[0]
        confidence = (1 / (1 + pow(2.71828, -abs(score)))) * 100
    else:
        confidence = 100.0  # No probability information available

    return prediction, round(float(confidence), 2)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # ---- POST: handle form submission ----
    title = (request.form.get("title") or "").strip()
    content = (request.form.get("content") or "").strip()

    # Basic validation
    if not title:
        flash("Please enter a news headline.", "error")
        return render_template("predict.html", title=title, content=content)

    if not content:
        flash("Please enter the news content.", "error")
        return render_template("predict.html", title=title, content=content)

    if len(content) < MIN_CONTENT_CHARACTERS:
        flash("Please provide more news content for meaningful analysis.", "error")
        return render_template("predict.html", title=title, content=content)

    if len(content) > MAX_CONTENT_CHARACTERS or len(title) > 500:
        flash(
            "The submitted content is too large to analyze. "
            "Please shorten the text and try again.",
            "error",
        )
        return render_template("predict.html", title=title, content=content)

    if model is None or vectorizer is None:
        flash(
            "The prediction model is not available yet. "
            "Please ask the administrator to run: python training/train_model.py",
            "error",
        )
        return render_template("predict.html", title=title, content=content)

    # ---- Run prediction ----
    try:
        prediction, confidence = predict_news(title, content)
    except Exception:
        traceback.print_exc()
        flash(
            "Something went wrong while analyzing the news content. "
            "Please try again.",
            "error",
        )
        return render_template("predict.html", title=title, content=content)

    # ---- Save to database (escape output for safe display later) ----
    safe_title = escape(title)
    safe_content = escape(content)

    try:
        save_prediction(safe_title, safe_content, prediction, confidence)
    except Exception:
        traceback.print_exc()
        flash(
            "The prediction was generated, but it could not be saved to the "
            "history database.",
            "warning",
        )

    return render_template(
        "result.html",
        title=safe_title,
        prediction=prediction,
        confidence=confidence,
        input_length=len(content),
    )


@app.route("/history")
def history():
    try:
        records = get_all_predictions()
        db_error = None
    except Exception:
        traceback.print_exc()
        records = []
        db_error = "Prediction history could not be loaded at this time."

    return render_template("history.html", records=records, db_error=db_error)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "GET":
        return render_template("contact.html")

    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    message = (request.form.get("message") or "").strip()

    email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    if not name:
        flash("Please enter your name.", "error")
        return render_template("contact.html", name=name, email=email, message=message)

    if not email or not re.match(email_pattern, email):
        flash("Please enter a valid email address.", "error")
        return render_template("contact.html", name=name, email=email, message=message)

    if not message:
        flash("Please enter your feedback message.", "error")
        return render_template("contact.html", name=name, email=email, message=message)

    try:
        save_feedback(escape(name), escape(email), escape(message))
        flash("Thank you for your feedback.", "success")
        return redirect(url_for("contact"))
    except Exception:
        traceback.print_exc()
        flash(
            "Your feedback could not be saved at this time. Please try again later.",
            "error",
        )
        return render_template("contact.html", name=name, email=email, message=message)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_error):
    traceback.print_exc()
    return render_template("500.html"), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=True is fine for LOCAL development only.
    # In production, this app is served with Gunicorn (see Procfile),
    # which does not use this __main__ block at all.
    app.run(host="0.0.0.0", port=port, debug=True)
