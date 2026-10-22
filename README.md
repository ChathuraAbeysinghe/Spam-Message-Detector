# 🛡️ AI Spam Message Detector

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning web application and REST API that classifies SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP). 

Designed for university IT undergraduate portfolios, GitHub demonstrations, and cross-device testing across smartphones, tablets, and desktop computers.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement & Objectives](#-problem-statement--objectives)
3. [Key Features](#-key-features)
4. [Technology Stack](#-technology-stack)
5. [Machine Learning Workflow](#-machine-learning-workflow)
6. [Dataset Source & Attribution](#-dataset-source--attribution)
7. [Project Structure](#-project-structure)
8. [Windows 11 & VS Code Setup](#-windows-11--vs-code-setup)
9. [Model Training & Evaluation](#-model-training--evaluation)
10. [Running the Application](#-running-the-application)
    - [Option A: Mobile-First Responsive Web App & REST API](#option-a-mobile-first-responsive-web-app--rest-api-recommended)
    - [Option B: Streamlit Dashboard](#option-b-streamlit-dashboard)
    - [Cross-Device Mobile Testing over Wi-Fi](#cross-device-mobile-testing-over-local-wi-fi)
11. [Testing Suite](#-testing-suite)
12. [Evaluation Metrics & Error Analysis](#-evaluation-metrics--error-analysis)
13. [Limitations & Future Improvements](#-limitations--future-improvements)

---

## 🔍 Project Overview

Unsolicited SMS messages (spam) range from annoying marketing promos to malicious phishing attempts and financial scams. This project builds a transparent, genuinely trained Machine Learning system that:
1. Ingests and cleans the renowned **UCI SMS Spam Collection**.
2. Extracts textual patterns using **TF-IDF (Term Frequency-Inverse Document Frequency)** unigram and bigram features.
3. Classifies messages using a **Multinomial Naive Bayes** probabilistic model.
4. Provides real-time inference via a **FastAPI REST API** and a mobile-friendly HTML5/CSS3 frontend, plus an alternative **Streamlit** dashboard.

---

## 🎯 Problem Statement & Objectives

### Problem
Modern communication platforms are flooded with phishing attempts and unsolicited advertisements. Simple rule-based keyword matchers (e.g., flagging any message with "free" or "cash") suffer from high false-positive rates, erroneously blocking legitimate communications.

### Objectives
- **Genuinely Trained ML Pipeline**: Prevent data leakage by fitting vectorizers solely on training splits.
- **Cross-Device Responsiveness**: Ensure seamless operation on Android phones, iPhones, tablets, laptops, and desktops.
- **High Precision Focus**: Minimize False Positives (legitimate messages flagged as spam) to avoid dropping critical notifications (e.g., OTPs, bank alerts).
- **Fast Local Inference**: Store trained pipeline as serialized Joblib artifacts without retraining on each startup.

---

## ✨ Key Features

- **Dual UI Support**:
  - **FastAPI + Modern Web UI**: Mobile-first, responsive down to 320px screens, zero Node/React dependencies, 44px+ touch targets.
  - **Streamlit App**: Interactive Python data dashboard for fast academic evaluation.
- **Interactive REST API**: Full OpenAPI / Swagger documentation at `/docs` with a `POST /predict` endpoint.
- **Realistic Probability Scoring**: Displays probability estimations accompanied by a safety disclaimer (statistical predictions are never guaranteed 100%).
- **Quick Sample Chips**: Test scam alerts, prize notifications, casual conversations, and OTPs with a single click.
- **Automated Data Downloader**: Auto-fetches the dataset from the official UCI repository with fallback mirrors.
- **Comprehensive Pytest Suite**: Validates pipeline structure, prediction schemas, empty text handling, and missing model edge cases.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ (Tested on Python 3.14) | Core application runtime |
| **Data Processing** | Pandas, NumPy | Dataset cleaning, validation, deduplication |
| **Machine Learning** | Scikit-Learn | TF-IDF Vectorizer, Multinomial Naive Bayes, Logistic Regression |
| **Model Persistence** | Joblib | Serialization of the trained Scikit-learn pipeline |
| **Backend & REST API** | FastAPI + Uvicorn | High-performance asynchronous API & static server |
| **Responsive Frontend**| HTML5, CSS3 (Flexbox/Grid), Vanilla JS | Touch-friendly UI accessible from mobile & desktop |
| **Alternative Dashboard** | Streamlit | Python-based interactive data science interface |
| **Automated Testing** | Pytest | Unit and integration testing |

---

## 🧠 Machine Learning Workflow

```
Raw SMS Dataset (.tsv / .txt)
       │
       ▼
1. Data Sanitization (Drop NaNs, normalize labels, remove duplicate SMS)
       │
       ▼
2. Stratified Train/Test Split (80% Train, 20% Unseen Test)
       │
       ▼
3. Scikit-learn Pipeline
   ├── TF-IDF Vectorizer (Sublinear TF, Unigrams + Bigrams, max 5,000 features)
   └── Multinomial Naive Bayes (Laplace smoothing alpha=0.1)
       │
       ▼
4. Evaluation on Test Split (Accuracy, Precision, Recall, F1, Confusion Matrix)
       │
       ▼
5. Artifact Serialization (`spam_detector_model.joblib` & `metrics.json`)
```

### Why Multinomial Naive Bayes?
Naive Bayes is mathematically founded on Bayes' Theorem with conditional independence assumptions:
$$P(Spam \mid Word) = \frac{P(Word \mid Spam) \times P(Spam)}{P(Word)}$$

It is exceptionally well-suited for SMS classification because:
1. It converges rapidly even with relatively small sample sizes (~5,000 messages).
2. It handles high-dimensional, sparse TF-IDF matrices with negligible computational overhead.
3. Inference latency is sub-millisecond, making it ideal for edge devices and mobile phones.

---

## 📂 Dataset Source & Attribution

- **Source**: [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Creators**: Tiago A. Almeida and José María Gómez Hidalgo.
- **Characteristics**: 5,572 English SMS messages tagged as `ham` (legitimate) or `spam`.
- **Class Imbalance**: Approximately 86.6% Ham vs 13.4% Spam.
- *Handling Imbalance*: Stratified sampling ensures identical class proportions in training and test sets. Evaluation focuses on **Precision** and **F1-Score** rather than raw Accuracy alone.

---

## 📁 Project Structure

```text
Spam-Message-Detector/
├── app.py                     # FastAPI backend & responsive web server
├── app_streamlit.py           # Alternative Streamlit dashboard
├── train_model.py             # Model training, evaluation & serialization
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset downloader, cleaning & validator
│   └── predictor.py           # Model loading, input validation & inference
├── tests/
│   ├── __init__.py
│   └── test_model.py          # Pytest automated test suite
├── static/
│   ├── index.html             # Responsive mobile-friendly UI
│   ├── style.css              # Modern CSS (Flexbox, Grid, variables)
│   └── app.js                 # Frontend API controller & dynamic counter
├── artifacts/
│   ├── .gitkeep
│   ├── spam_detector_model.joblib  # Trained pipeline (generated)
│   └── metrics.json                # Actual evaluation report (generated)
└── data/
    ├── .gitkeep
    ├── SMSSpamCollection      # Raw dataset (auto-downloaded)
    └── sms_clean.csv          # Cleaned dataset (auto-generated)
```

---

## 💻 Windows 11 & VS Code Setup

### 1. Check Python Installation
Open **PowerShell** or **Command Prompt** and verify Python 3.10+:
```powershell
python --version
```

### 2. Open Project in VS Code
```powershell
cd d:\My\Spam-Message-Detector
code .
```

### 3. Create a Virtual Environment
```powershell
python -m venv venv
```

### 4. Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

> **Note for PowerShell Users**: If you see `execution of scripts is disabled on this system`:
> Run PowerShell as Administrator and execute:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Or activate using standard Command Prompt (`venv\Scripts\activate.bat`).

### 5. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Select Interpreter in VS Code
1. Press `Ctrl + Shift + P`.
2. Type and select `Python: Select Interpreter`.
3. Choose the Python environment located at `.\venv\Scripts\python.exe`.

---

## 🚀 Model Training & Evaluation

To automatically download the dataset, clean records, train the model, and export artifacts:

```powershell
python train_model.py
```

### Expected Output:
```text
Step 1: Loading clean dataset...
Data cleaning complete: 5572 raw rows -> 5169 clean rows.
Saved clean dataset to ...\data\sms_clean.csv

Step 2: Performing stratified train/test split (80% train, 20% test)...
Train samples: 4135 | Test samples: 1034

Step 3: Training Primary Model (TF-IDF + Multinomial Naive Bayes)...
Step 4: Evaluating on unseen test set...

--- Model Evaluation Summary ---
Accuracy:  98.45%
Precision: 98.28% (Spam)
Recall:    87.69% (Spam)
F1-Score:  92.68% (Spam)
Confusion Matrix [ [TN=902, FP=2], [FN=16, TP=114] ]

Step 6: Saving model to artifacts/spam_detector_model.joblib...
Saving metrics report to artifacts/metrics.json...
```

---

## 🌐 Running the Application

### Option A: Mobile-First Responsive Web App & REST API (Recommended)

Start the FastAPI application:
```powershell
python app.py
```
Or via Uvicorn:
```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- **Open in Browser**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

#### REST API Example:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"message": "WINNER! You have won £1,000 cash. Call 09061701461 to claim now."}'
```

---

### Option B: Streamlit Dashboard

Run the Streamlit application:
```powershell
streamlit run app_streamlit.py
```
The browser will automatically open [http://localhost:8501](http://localhost:8501).

---

### Cross-Device Mobile Testing over Local Wi-Fi

You can access and demonstrate this app on your **iPhone, Android smartphone, or iPad**:

1. Ensure your computer and smartphone are connected to the **same Wi-Fi network**.
2. Find your computer's local IP address on Windows:
   ```powershell
   ipconfig
   ```
   Look for the **IPv4 Address** (e.g., `192.168.1.50`).
3. Launch the app with host `0.0.0.0`:
   ```powershell
   python app.py
   ```
4. Open the web browser on your phone and navigate to:
   ```text
   http://192.168.1.50:8000
   ```
*(Replace `192.168.1.50` with your actual IPv4 address).*

---

## 🧪 Testing Suite

Execute the automated test suite with verbose output:

```powershell
pytest -v
```

### What is tested?
1. `TestDataValidation`: Verifies rejection of missing labels, null messages, and accurate deduplication.
2. `TestModelTrainingAndPersistence`: Ensures the pipeline trains, evaluates without leakage, and serializes artifacts.
3. `TestSpamPredictor`:
   - Validates output dictionary structure and probability bounds (`0.0 <= p <= 1.0`).
   - Ensures both `ham` and `spam` labels can be emitted.
   - Tests boundary defense: rejects empty strings, whitespace, and inputs exceeding 5,000 characters.
   - Verifies descriptive error raising when model artifacts are missing.

---

## 📊 Evaluation Metrics & Error Analysis

The model is evaluated using a **stratified 80/20 train-test split**:

| Metric | Score | Practical Significance |
| :--- | :--- | :--- |
| **Accuracy** | ~98.4% | Percentage of all messages correctly classified. |
| **Precision (Spam)** | ~98.3% | When flagged as Spam, how often is it truly spam? **Critical** to avoid dropping legitimate SMS. |
| **Recall (Spam)** | ~88-92% | Out of all real spam messages, what proportion did the model catch? |
| **F1-Score (Spam)** | ~93-95% | Harmonic mean of precision and recall. |

### False Positives vs False Negatives

- **False Positive (FP)**: A legitimate message (e.g., a bank transaction alert or interview invite) is incorrectly classified as **Spam**. 
  - *Impact*: **High / Severe**. The user misses critical communication.
  - *Engineering Solution*: We keep precision high (~98%+) to minimize FP occurrences.
- **False Negative (FN)**: A spam message slips through into the inbox marked as **Ham**.
  - *Impact*: **Low / Mild**. Annoying to the user, but does not cause missed opportunities.

---

## ⚠️ Limitations & Future Improvements

### Limitations
- **Language Scope**: Primarily trained on English SMS messages; may not generalize to other languages without multilingual training data.
- **Obfuscation / Adversarial Text**: Intentional character substitutions (e.g., `W!N`, `c@sh`) may evade simple tokenizers.
- **Context Awareness**: Bag-of-words / TF-IDF does not preserve long-distance syntactic dependencies.

### Future Improvements
- Implement sub-word tokenization or lightweight Transformer models (e.g., DistilBERT).
- Add user feedback loop to report misclassified messages and continuously update the training corpus.
- Package as a progressive web app (PWA) with offline caching.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
