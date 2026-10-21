# Machine Learning Framework for Customer Sentiment Analysis from Social Media Data

A modular, extensible Python machine-learning framework designed to ingest, validate, preprocess, and classify customer sentiment (Positive, Negative, Neutral) from social media and customer review data.

---

## 📌 Project Architecture & Workflow

```text
RAW TEXT DATA
      ↓
DATA INGESTION (Flexible column mapping)
      ↓
DATA VALIDATION (Quality checks, deduplication, schema check)
      ↓
TEXT PREPROCESSING (Cleaning, tokenization, lemmatization, negation-preserving)
      ↓
EXPLORATORY DATA ANALYSIS (Distributions, n-grams, trends)
      ↓
TF-IDF FEATURE EXTRACTION (Word unigrams/bigrams, fit on train split only)
      ↓
MODEL TRAINING & COMPARISON (MultinomialNB, Logistic Regression, Linear SVM)
      ↓
MODEL EVALUATION & ERROR ANALYSIS (Macro-F1, Confusion Matrix, Misclassifications)
      ↓
SENTIMENT PREDICTION PIPELINE (Single & batch inference, predicted probabilities)
      ↓
INSIGHT & ANALYTICS ENGINE (Trends, complaints, sentiment alerts)
      ↓
INTERACTIVE STREAMLIT DASHBOARD (Data inspection, exploration, live prediction)
```

---

## 📂 Project Structure

```text
customer-sentiment-framework/
├── data/
│   ├── raw/                 # Raw dataset files
│   ├── processed/           # Validated and preprocessed datasets
│   └── sample/              # Sample datasets for quick verification
├── notebooks/               # Interactive exploration notebooks
│   ├── 01_eda.ipynb
│   └── 02_model_experiments.ipynb
├── src/                     # Core framework source code
│   ├── __init__.py
│   ├── data_loader.py       # Ingestion and standardization
│   ├── validation.py        # Validation and quality reporting
│   ├── preprocessing.py     # NLP cleaning and tokenization
│   ├── features.py          # TF-IDF feature extraction
│   ├── train.py             # Model training pipelines
│   ├── evaluate.py          # Metrics, confusion matrix, error analysis
│   ├── predict.py           # Inference engine for new texts
│   └── insights.py          # Business KPIs and trend analytics
├── models/                  # Serialized models and vectorizers
├── dashboard/               # Streamlit interactive user interface
│   └── app.py
├── tests/                   # Pytest test suite
│   ├── __init__.py
│   ├── test_foundation.py
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── requirements.txt         # Core dependencies
├── README.md                # Project documentation and setup guide
└── .gitignore
```

---

## 🚀 Setup Instructions

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests:**
   ```bash
   pytest
   ```
