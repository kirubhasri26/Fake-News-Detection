# AI-Based Fake News Detection System

A beginner-friendly, full-stack web application that uses Natural Language
Processing (NLP) and Machine Learning to predict whether a piece of news
content is likely **REAL** or **FAKE**.

---

## 1. Description

This project analyzes a news headline and article body using a
**TF-IDF Vectorizer + Logistic Regression** model, and returns a prediction
(REAL/FAKE) along with a confidence score. It also stores every prediction
in a local SQLite database so you can review your prediction history.

This is a **college / educational demo project**, not a production
fact-checking tool.

---

## 2. Features

- Home page with project overview
- News analysis form (headline + content)
- ML-based prediction with confidence score
- Prediction history stored in SQLite
- About page explaining the project and technologies
- Contact / feedback form
- Friendly error handling (no raw Python errors shown to users)
- Responsive, mobile-friendly Bootstrap UI
- Ready for free deployment (Render, or similar platforms)

---

## 3. Technologies

| Layer        | Technology                              |
|--------------|------------------------------------------|
| Frontend     | HTML5, CSS3, JavaScript, Bootstrap 5     |
| Backend      | Python, Flask                            |
| Machine Learning | scikit-learn (TF-IDF + Logistic Regression), pandas, numpy, joblib |
| Database     | SQLite                                   |
| Deployment   | Gunicorn + Render (or similar free host) |

---

## 4. Project Structure

```
fake_news_detection/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md                # This file
├── Procfile                 # Deployment start command (Gunicorn)
├── .gitignore
├── runtime.txt               # Python version for deployment
├── .env.example              # Example environment variables
│
├── model/
│   ├── model.pkl              # Trained Logistic Regression model
│   └── vectorizer.pkl         # Trained TF-IDF vectorizer
│
├── dataset/
│   └── news.csv                # Demo training dataset (title, text, label)
│
├── training/
│   └── train_model.py          # Script that trains and saves the model
│
├── database/
│   ├── __init__.py
│   └── database.py             # SQLite helper functions
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── history.html
│   ├── about.html
│   ├── contact.html
│   ├── 404.html
│   └── 500.html
│
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
│
└── instance/
    └── database.db          # Created automatically when the app runs
```

### What each important file does

- **app.py** — the Flask web server. Defines all routes (`/`, `/predict`,
  `/history`, `/about`, `/contact`), loads the ML model, validates input,
  runs predictions, and saves results to the database.
- **training/train_model.py** — reads `dataset/news.csv`, cleans the text,
  trains the TF-IDF + Logistic Regression model, prints accuracy, and saves
  `model/model.pkl` and `model/vectorizer.pkl`.
- **database/database.py** — creates the SQLite tables (`predictions`,
  `feedback`) and provides functions to save/read data.
- **templates/** — all the HTML pages, using Jinja2 templating and a shared
  `base.html` layout (navbar + footer).
- **static/css/style.css** — the visual design (colors, spacing, cards).
- **static/js/script.js** — shows a loading message while a prediction is
  being processed.
- **dataset/news.csv** — a **small demo dataset** (80 rows) created only for
  demonstration. See the "Limitations" section below.

---

## 5. Requirements

- Python 3.10+ installed on your computer
- VS Code (recommended) or any code editor
- Basic internet connection (for installing packages and CDN assets)

Check your Python version:

```bash
python --version
```

---

## 6. Installation (Windows + VS Code)

Open the project folder in VS Code, then open a terminal (`Terminal > New
Terminal`) and run the following commands **in order**.

### Step 1 — Create a virtual environment

```bash
python -m venv venv
```

*This creates an isolated Python environment for this project, so its
packages don't conflict with anything else on your computer.*

### Step 2 — Activate the virtual environment

```bash
venv\Scripts\activate
```

*This activates the project's environment. You should see `(venv)` appear
at the start of your terminal line.*

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

*This installs Flask, pandas, numpy, scikit-learn, joblib, and gunicorn —
all the Python libraries the project needs.*

---

## 7. Train the Model

Before running the app for the first time (or any time you delete the
`model/` folder), train the machine-learning model:

```bash
python training/train_model.py
```

You should see output similar to:

```
Training started...
Dataset loaded... (80 rows)
Training completed...
Accuracy: XX.XX%
Model saved successfully.
```

This creates `model/model.pkl` and `model/vectorizer.pkl`.

---

## 8. Run the Application

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

To stop the server, press `CTRL + C` in the terminal.

To leave the virtual environment when you're done:

```bash
deactivate
```

---

## 9. Testing Checklist

| # | Test                          | Expected Result                          |
|---|-------------------------------|-------------------------------------------|
| 1 | Open homepage                 | Homepage loads correctly                   |
| 2 | Submit empty news              | Validation message is shown                |
| 3 | Submit a valid headline + content | REAL/FAKE prediction with confidence shown |
| 4 | Open Prediction History        | Previous prediction appears in the table  |
| 5 | Refresh the page               | Application remains functional             |
| 6 | Open About page                | About page loads with project details      |
| 7 | Open Contact page              | Feedback form works and shows thank-you    |
| 8 | Visit an invalid URL           | A friendly 404 page is shown               |

---

## 10. GitHub Setup

1. Create a free account at https://github.com if you don't have one.
2. Create a new (empty) repository on GitHub — do not add a README there.
3. In your project folder, run:

```bash
git init
git add .
git commit -m "Initial project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL GitHub shows you after
creating the repository (for example:
`https://github.com/your-username/fake-news-detection.git`).

---

## 11. Free Deployment (Render)

> Render's exact plan names and free-tier limits can change over time.
> Check Render's current pricing/docs page before deploying, since free
> web-service availability and behavior can change.

1. Create a free account at https://render.com and sign in.
2. Click **New +** → **Web Service**.
3. Connect your GitHub account and select this repository.
4. Configure the service:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Add an environment variable `FLASK_SECRET_KEY` with a long random value
   (Render lets you set environment variables in the dashboard).
6. Click **Create Web Service** / **Deploy**.
7. Once the build finishes, open the generated `.onrender.com` URL to view
   your live site.

**Important:** Make sure `model/model.pkl` and `model/vectorizer.pkl` are
committed to your GitHub repository (they are small files), since the
free hosting environment will not run `train_model.py` automatically.

---

## 12. Important Limitation: SQLite on Free Hosting

Many free hosting platforms (including Render's free tier) use
**ephemeral storage** — meaning any files written while the app is
running (including the SQLite database) may be **deleted whenever the
service restarts or redeploys**. For a college demo, this is acceptable:
your Prediction History will work while the app is running, but it may
reset after the free instance sleeps, restarts, or is redeployed. For
permanent history storage in a real production system, you would use a
managed database service instead of SQLite.

---

## 13. Limitations

1. The model depends on the quality and coverage of its training dataset.
2. The system analyzes textual patterns rather than independently
   verifying facts.
3. Confidence scores represent model confidence, not factual certainty.
4. Newly emerging topics may not be well represented in the training data.
5. The system should not replace professional fact-checking or trusted
   news sources.
6. **`dataset/news.csv` is a small DEMO dataset (80 rows)** created only so
   the project can be trained and demonstrated end-to-end. It is **not** a
   real-world dataset, and its accuracy figures should not be treated as
   representative of real-world performance. For a stronger model, replace
   it with a larger, real-world labeled dataset (for example, a public
   "Fake and Real News" dataset from Kaggle) that has the same column
   names: `title`, `text`, `label`.

---

## 14. Future Enhancements

- Larger and more diverse datasets
- Real-time news verification
- Source credibility analysis
- Explainable AI techniques (e.g., highlighting influential words)
- Multilingual fake news detection
- Transformer-based NLP models (e.g., BERT)
- Integration with trusted fact-checking databases
- Advanced deep-learning models

---

## 15. Troubleshooting

| Error | Solution |
|---|---|
| `'python' is not recognized` | Install Python from python.org and make sure "Add Python to PATH" is checked during installation. Restart your terminal. |
| `'pip' is not recognized` | Use `python -m pip install -r requirements.txt` instead, or reinstall Python with pip included. |
| `ModuleNotFoundError` | Make sure your virtual environment is activated (`venv\Scripts\activate`), then run `pip install -r requirements.txt` again. |
| `model.pkl not found` / model load message | Run `python training/train_model.py` to train and save the model files. |
| `TemplateNotFound` | Make sure you are running `python app.py` from the project's root folder, not from inside a subfolder. |
| `Port already in use` | Close any other program using port 5000, or set a different port with `set PORT=5001` (Windows) before running `python app.py`. |
| `'git' is not recognized` | Install Git from https://git-scm.com and restart your terminal. |
| Deployment failed on Render | Check the Render build logs for the exact error, confirm `requirements.txt` and the start command are correct, and that `model/model.pkl` and `model/vectorizer.pkl` were committed to GitHub. |

---

## 16. Simple Explanation of Key Terms

- **Frontend** — the part of the website users see and interact with
  (HTML, CSS, JavaScript).
- **Backend** — the server-side logic that processes requests and returns
  responses (here, Flask).
- **Flask** — a lightweight Python web framework used to build the
  website and its routes.
- **Machine Learning** — teaching a computer to recognize patterns from
  examples (data) instead of writing explicit rules.
- **NLP (Natural Language Processing)** — a field of AI focused on
  helping computers understand and process human language/text.
- **TF-IDF** — a technique that converts text into numbers by measuring
  how important a word is to a document relative to a collection of
  documents.
- **Logistic Regression** — a simple, widely-used machine-learning
  algorithm for classification tasks (like REAL vs FAKE).
- **SQLite** — a lightweight, file-based database that requires no
  separate server, ideal for small projects.
- **model.pkl** — the trained Logistic Regression model, saved to disk
  using `joblib` so it can be reused without retraining.
- **vectorizer.pkl** — the trained TF-IDF vectorizer, saved so the same
  text-to-numbers transformation can be applied to new input.
- **API / Route** — a URL endpoint (like `/predict`) that the frontend
  calls to trigger backend logic.
- **Deployment** — the process of making your application available on
  the internet so others can access it.
- **GitHub** — a platform for storing and sharing code using Git version
  control.
- **Render** — a cloud platform that can host and run web applications,
  including free-tier options suitable for small projects.

---

## 17. College Project Content

### Abstract
This project presents an AI-based system for detecting fake news using
Natural Language Processing and Machine Learning. The system analyzes the
textual content of a news article and classifies it as Real or Fake using
a TF-IDF and Logistic Regression pipeline, presenting results through a
web-based interface built with Flask.

### Problem Statement
The rapid spread of misinformation through digital platforms makes it
difficult for readers to distinguish credible news from fabricated
content. Manual fact-checking cannot keep pace with the volume of content
being published and shared every day.

### Objectives
- To build a web application that accepts news content as input.
- To apply NLP techniques to preprocess and vectorize the text.
- To train a machine-learning model that classifies news as Real or Fake.
- To display prediction results with a confidence score.
- To maintain a history of previous predictions.

### Existing System
Existing fact-checking approaches largely rely on manual verification by
journalists and fact-checking organizations, which is accurate but slow
and cannot scale to the volume of online content.

### Proposed System
The proposed system uses a machine-learning model trained on labeled news
examples to automatically flag content that shares textual patterns with
known fake or real news, providing a fast, automated first-pass
classification.

### System Requirements
A computer capable of running Python 3.10+, a modern web browser, and
(optionally) an internet connection for deployment.

### Software Requirements
Python, Flask, scikit-learn, pandas, numpy, joblib, SQLite, HTML/CSS/
JavaScript, Bootstrap.

### Hardware Requirements
Any standard computer with at least 4GB RAM is sufficient to run and
demonstrate this project locally.

### Modules
1. User Interface Module (HTML/CSS/JS templates)
2. Text Preprocessing Module
3. Machine Learning Module (TF-IDF + Logistic Regression)
4. Database Module (SQLite)
5. Flask Application/Routing Module

### System Architecture
Browser (Frontend) ⇄ Flask Application (Backend) ⇄ ML Model (TF-IDF +
Logistic Regression) ⇄ SQLite Database

### Data Flow
User enters news → Flask receives form data → text is cleaned and
combined → TF-IDF transforms text into numeric features → Logistic
Regression predicts REAL/FAKE with a confidence score → the result is
saved to SQLite → the result is displayed to the user.

### Algorithm
1. Collect and clean the dataset.
2. Combine title and article text.
3. Split into training and testing sets.
4. Convert text into TF-IDF feature vectors.
5. Train a Logistic Regression classifier on the training set.
6. Evaluate accuracy on the test set.
7. Save the trained model and vectorizer for later use in the web app.

### Implementation
Implemented using Python and Flask for the backend, scikit-learn for the
ML pipeline, SQLite for persistence, and Bootstrap-based HTML/CSS for the
frontend.

### Testing
See the Testing Checklist section above; all core routes, form
validations, and the prediction pipeline were manually verified.

### Advantages
Fast, automated, easy to deploy, requires no paid services, and provides
an interpretable confidence score alongside each prediction.

### Limitations
See the Limitations section above.

### Future Enhancement
See the Future Enhancements section above.

### Conclusion
This project demonstrates how NLP and Machine Learning can be combined
with a web application to build a functional, educational fake news
detection tool, while highlighting the importance of dataset quality and
the limits of automated fact-checking.

---

## 18. Viva Preparation (Sample Questions & Answers)

1. **What is fake news?**
   False or misleading information presented as legitimate news content.
2. **Why did you choose this project?**
   Misinformation is a growing real-world problem, and this project
   demonstrates practical NLP and ML skills applied to it.
3. **What is machine learning?**
   A method where computers learn patterns from data instead of being
   explicitly programmed with rules.
4. **Why use NLP?**
   News content is unstructured text; NLP techniques convert it into a
   form a machine-learning model can process.
5. **What is TF-IDF?**
   A technique that converts text into numeric vectors by weighing words
   based on their frequency in a document versus the entire dataset.
6. **Why Logistic Regression?**
   It is a simple, fast, and interpretable algorithm well-suited to
   binary text classification tasks like REAL vs FAKE.
7. **What is training data?**
   The portion of labeled data used to teach the model patterns.
8. **What is testing data?**
   A separate portion of labeled data used to evaluate how well the
   trained model generalizes to unseen examples.
9. **What is accuracy?**
   The percentage of predictions the model got correct on the test data.
10. **What is Flask?**
    A lightweight Python web framework used to build the application's
    backend and routes.
11. **Why SQLite?**
    It is a simple, file-based database that requires no separate server,
    making it ideal for small projects and demos.
12. **What is deployment?**
    Making the application accessible on the internet, typically on a
    hosting platform such as Render.
13. **What is GitHub?**
    A platform for storing code with version control, and connecting to
    hosting platforms for deployment.
14. **What are the limitations of this project?**
    It relies on a small demo dataset, analyzes patterns rather than
    verified facts, and its confidence score reflects model certainty,
    not factual truth.
15. **Can the system guarantee news is true?**
    No. It provides a statistical prediction based on learned patterns,
    not a factual verification.
16. **Can it detect newly emerging fake news?**
    Only to the extent that its patterns resemble what the model was
    trained on; genuinely novel misinformation styles may not be
    detected reliably.
17. **How can this project be improved?**
    By using a larger real-world dataset, more advanced models (e.g.,
    transformers), and source credibility signals.
18. **What does the confidence score mean?**
    The probability the model assigns to its predicted class, reflecting
    how strongly the input matches learned patterns.
19. **What is joblib used for?**
    To save and load the trained model and vectorizer to/from disk
    efficiently.
20. **What is a route in Flask?**
    A URL pattern (e.g., `/predict`) mapped to a Python function that
    handles requests to that URL.

---

## 19. Final Notes

- Local URL: `http://127.0.0.1:5000`
- Always retrain the model (`python training/train_model.py`) if you
  update `dataset/news.csv`.
- Never commit real secrets — use `.env.example` as a template and keep
  actual secrets out of GitHub.
