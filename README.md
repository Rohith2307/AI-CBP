# Machine Learning Framework for Customer Sentiment Analysis from Social Media Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-82%20passed-brightgreen.svg)](tests/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](data/README.md)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](dashboard/app.py)

A modular, production-ready Python machine learning framework that automates customer sentiment classification (Positive, Negative, Neutral), validates data hygiene, identifies operational grievance spikes, and translates unstructured feedback into measurable business intelligence.

---

## 📌 System Architecture & Pipeline Workflow

```text
                     RAW CUSTOMER TEXT DATA
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION & AUTO-DETECTION (src/data_loader.py)     │
│    • Flexible column mapping (aliases for text, date, etc.) │
│    • Robust encoding fallback (UTF-8, Latin-1, CP1252)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA VALIDATION & AUDITING (src/validation.py)           │
│    • Deduplication (purged 213 duplicate tweets)            │
│    • Missing/empty text checks & label normalization        │
│    • Structured audit logging without silent row deletion   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TEXT PREPROCESSING (src/preprocessing.py)                │
│    • Preserves original text column for human inspection    │
│    • Converts emojis to tokens (emoji_pos, emoji_neg)       │
│    • Preserves negations (not, never, n't) & expands slurs  │
│    • NLTK tokenization & WordNet lemmatization              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. EXPLORATORY DATA ANALYSIS (src/eda.py)                   │
│    • Class distribution (62.94% Neg, 21.21% Neu, 15.85% Pos)│
│    • Character & word length distributions by sentiment     │
│    • N-gram frequency & top complaint keyword extraction    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ANTI-LEAKAGE FEATURE EXTRACTION (src/features.py)        │
│    • Stratified 70% Train / 15% Val / 15% Test split        │
│    • TF-IDF word unigrams + bigrams (max 5,000 features)    │
│    • STRICT: Vectorizer fitted ONLY on Train split          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. MODEL SELECTION & TRAINING (src/train.py)                │
│    • Candidates: Logistic Regression, Linear SVM, MNB       │
│    • 5-Fold Stratified Cross-Validation on Train split       │
│    • Selection via Validation Macro F1 (Logistic Regression)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. UNBIASED TEST EVALUATION (src/evaluate.py)               │
│    • Strict evaluation on 2,165 held-out test samples       │
│    • 77.97% Accuracy | 0.7329 Macro F1                      │
│    • Confusion Matrix & systematic error archetype audit    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. INFERENCE & ANALYTICS ENGINE (src/predict.py, insights.py│
│    • Real-time single & batch classification with proba     │
│    • Net Sentiment Score (NSS) & negative grievance spikes  │
│    • Methodological causality disclaimers                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. INTERACTIVE STREAMLIT DASHBOARD (dashboard/app.py)       │
│    • 8 dynamic pages: Overview, Ingestion, EDA, Inference,  │
│      Explorer, Trends & Spikes, Model Audit, Export         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
AI CBP/
├── data/
│   ├── raw/                           # Raw input dataset
│   │   └── airline_sentiment_raw.csv  # Full Crowdflower Twitter Airline dataset (14,640 rows)
│   ├── processed/                     # Validated & cleaned datasets
│   │   └── validated_feedback.csv     # 14,427 validated records
│   ├── sample/                        # Fast local test fixtures
│   │   └── sample_airline_sentiment.csv # 300 balanced rows (100 Pos / 100 Neg / 100 Neu)
│   └── README.md                      # Dataset provenance, license, and schema documentation
├── models/
│   ├── sentiment_model.pkl            # Final deployment Pipeline (TF-IDF + Logistic Regression)
│   ├── vectorizer.pkl                 # Fitted TfidfVectorizer artifact
│   ├── model_comparison.json          # Stage 7 candidate comparison metrics & CV scores
│   └── evaluation_results.json        # Stage 8 held-out test evaluation & error audit
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis walkthrough
│   └── 02_model_experiments.ipynb     # Model training and hyperparameter experiments
├── src/
│   ├── __init__.py                    # Source package marker
│   ├── data_loader.py                 # File ingestion, schema auto-detection & encoding fallback
│   ├── validation.py                  # Data validation, deduplication & audit reporting
│   ├── preprocessing.py               # Negation-preserving NLP cleaning & lemmatization
│   ├── features.py                    # Stratified splitting & anti-leakage TF-IDF extraction
│   ├── train.py                       # Candidate model training, CV & pipeline packaging
│   ├── evaluate.py                    # Test split metrics, confusion matrix & error analysis
│   ├── predict.py                     # Unified inference engine (single & batch)
│   └── insights.py                    # Business KPIs, spike detection & report generation
├── dashboard/
│   └── app.py                         # 8-page interactive Streamlit dashboard
├── tests/
│   ├── test_foundation.py             # Environment & import verification
│   ├── test_data_loader.py            # Ingestion & auto-mapping tests
│   ├── test_validation.py             # Validation & deduplication tests
│   ├── test_preprocessing.py          # NLP tokenization & negation preservation tests
│   ├── test_features.py               # Anti-leakage splitting & TF-IDF tests
│   ├── test_train.py                  # Training pipeline & candidate comparison tests
│   ├── test_evaluate.py               # Metric calculation & error audit tests
│   ├── test_prediction.py             # Single & batch inference tests
│   ├── test_insights.py               # KPI calculation & spike detection tests
│   ├── test_dashboard.py              # Streamlit AppTest navigation & interaction tests
│   └── test_end_to_end_workflow.py    # Full pipeline end-to-end integration test
├── .streamlit/
│   ├── config.toml                    # Headless server & port configuration
│   └── credentials.toml               # Non-interactive credentials setup
├── requirements.txt                   # Production dependencies
└── README.md                          # Framework documentation
```

---

## ⚡ Setup & Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone & Navigate to Repository
```bash
git clone <repository-url>
cd "AI CBP"
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Required NLTK Corpora
```bash
python -c "
import nltk
for res in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(res, quiet=True)
print('NLTK resources downloaded successfully.')
"
```

---

## 🎯 Verification & Run Commands

### Run the Automated Test Suite (82 Tests)
Run all unit and integration tests across every module:
```bash
pytest tests/ -v
```

### Run the End-to-End Pipeline Verification Test
```bash
pytest tests/test_end_to_end_workflow.py -v
```

### Launch the Streamlit Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📊 Dataset Provenance & Data Hygiene

- **Source:** *Twitter US Airline Sentiment* dataset (Crowdflower / Kaggle, CC BY-NC-SA 4.0).
- **Raw Volume:** 14,640 records across 6 major US airlines.
- **Validation Audit:**
  - **Duplicate comments detected & purged:** 213 rows (exact duplicate tweets from retweets/bots).
  - **Empty or missing comments removed:** 0 rows.
  - **Retained clean records:** 14,427 rows.
- **Class Balance (Severe Real-World Imbalance):**
  - **Negative:** 9,079 (62.94%)
  - **Neutral:** 3,060 (21.21%)
  - **Positive:** 2,288 (15.85%)
  - **Imbalance Ratio:** 3.97:1 (Negative to Positive). Addressed via `class_weight='balanced'` in training.

---

## 🧪 Model Experiments & Measured Benchmark Results

### 1. Candidate Comparison (Stage 7)
All models were trained strictly on the 70% Training Split (10,098 samples) and compared using **5-Fold Cross-Validation** and the **15% Validation Split** (2,164 samples).

| Model Candidate | Validation Accuracy | Validation Macro F1 | CV Macro F1 (Train) | Inference Latency | Supports Probabilities |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (balanced)** | **77.50%** | **0.7217** | **0.7461 ± 0.0044** | **0.44 ms** | **Yes** |
| Linear SVM (`LinearSVC`) | 78.37% | 0.7179 | 0.7350 ± 0.0049 | 0.40 ms | No (calibrated decision) |
| Multinomial Naive Bayes | 74.08% | 0.6004 | 0.6174 ± 0.0071 | 0.49 ms | Yes |

> **Selection Decision:** **Logistic Regression** was chosen as the production model because it achieved the highest Validation Macro F1 (0.7217) and highest 5-fold CV score (0.7461), correctly balances minority positive/neutral recall via inverse class weights, and natively produces calibrated class probabilities.

---

### 2. Final Test Set Evaluation (Stage 8)
Evaluated strictly **once** on the 15% Held-Out Test Set (2,165 samples) that was **never exposed** to training, feature fitting, or model selection.

- **Test Accuracy:** **77.97%**
- **Test Macro F1:** **0.7329**
- **Test Macro Precision:** **0.7169**
- **Test Macro Recall:** **0.7560**
- **Test Weighted F1:** **0.7860**

#### Detailed Per-Class Classification Report:
| Sentiment Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 0.9061 | 0.8150 | 0.8581 | 1,362 |
| **Neutral** | 0.5746 | 0.6797 | 0.6228 | 459 |
| **Positive** | 0.6700 | 0.7733 | 0.7179 | 344 |

#### Test Set Confusion Matrix ($N = 2,165$):
```text
                     Predicted Negative   Predicted Neutral   Predicted Positive
Actual Negative :          1,110                 184                  68
Actual Neutral  :             84                 312                  63
Actual Positive :             31                  47                 266
```

---

## 🔍 Error Analysis & Failure Archetypes

Systematic audit of the 477 misclassifications in the held-out test set revealed 6 distinct linguistic error archetypes:

1. **Semantic Ambiguity (272 errors, 57.0%):** Subtle wording where neutral inquiries touch on negative airline events (e.g., *"Is flight 412 cancelled?"*).
2. **Complex Negation (154 errors, 32.3%):** Multi-clause sentences where negations flip polarity across long contexts (e.g., *"Not bad, but could be much better"*).
3. **Sarcasm & Irony (16 errors, 3.4%):** Positive praise tokens used to mock severe operational failures (e.g., *"Thanks JetBlue for leaving my bags in Boston! Fantastic job."*).
4. **Short / Low Context (15 errors, 3.1%):** Terse comments (e.g., *"seriously?"*, *"wow"*) lacking explicit sentiment keywords.
5. **Ambiguous Inquiries (14 errors, 2.9%):** Customer service questions that mimic complaints.
6. **Mixed Sentiment (6 errors, 1.3%):** Dual-sentiment comments containing equal praise and complaint (e.g., *"Great pilots, terrible ground staff"*).

---

## ⚠️ Limitations & Responsible AI Guidelines

1. **No Absolute Accuracy Guarantee:** Sentiment analysis on informal social media contains inherent human annotation ambiguity; accuracy is bounded around 78–82% for discrete 3-class models.
2. **Sarcasm Limitation:** N-gram TF-IDF models cannot infer vocal tone or situational irony when praise keywords are used sarcastically.
3. **Domain Vocabulary Specificity:** The current model is trained on airline customer communications. Classifying medical, financial, or e-commerce reviews requires fine-tuning or domain-specific retraining.
4. **Causality vs. Correlation Disclaimer:** An identified negative grievance spike (e.g., during severe weather) reflects elevated negative volume, but does not prove operational negligence. Human root-cause verification is essential.

---

## 🚀 Implemented Features vs. Planned Extensions

| Dimension | Implemented in Current Framework (Stages 1–11) | Planned Future Scope (Post-MVP) |
| :--- | :--- | :--- |
| **Model Families** | Logistic Regression (balanced), Linear SVM, Multinomial Naive Bayes | Transformer models (RoBERTa, DeBERTa-v3) |
| **Feature Extraction** | Word unigrams + bigrams TF-IDF (strict train-only fit) | Dense contextual sentence embeddings |
| **Classification** | Discrete 3-class (Positive, Negative, Neutral) + probabilities | Aspect-Based Sentiment Analysis (ABSA) |
| **Data Ingestion** | CSV upload with auto-mapping & Latin-1/UTF-8 fallback | Real-time Twitter/X API streaming webhooks |
| **Spike Detection** | Leave-one-out statistical negative spike alert with volume gates | Dynamic seasonality-aware Bayesian anomaly detection |
| **Deployment** | Standalone interactive Streamlit dashboard & serializable joblib pipelines | Containerized FastAPI microservice with Docker & Kubernetes |
| **Testing** | 82 automated unit, integration, and UI tests via pytest | End-to-end Cypress UI browser automation |

---

## 🎬 3-Minute Live Demonstration Script

1. **Overview (0:00 - 0:30):**
   - Open dashboard at `http://localhost:8501`.
   - Point out the active KPIs: Total Comments (14,427), Negative Rate (62.9%), Positive Rate (15.9%), and Net Sentiment Score (-47.1).
   - Show the dynamic class distribution and donut charts.
2. **Ingestion & Quality Audit (0:30 - 1:00):**
   - Navigate to **📂 Ingestion & Quality**.
   - Show the duplicate detection metric: 213 duplicate tweets automatically caught and purged.
   - Expand the **Audited Removal Log** to display exact purged duplicates with human-readable audit reasons.
3. **Exploratory Data Analysis (1:00 - 1:30):**
   - Navigate to **📊 Exploratory Data (EDA)**.
   - Show that negative feedback averages 112 characters (users explain specific grievances) vs positive feedback at 85 characters.
   - Inspect top complaint keywords (*delayed*, *hours*, *cancelled*, *customer service*).
4. **Live Inference (1:30 - 2:00):**
   - Navigate to **🧠 Sentiment Analysis**.
   - Select the sarcastic prompt: *"Delayed 5 hours, no food vouchers, and lost our bags. Worst experience ever! 😡"*.
   - Click **Predict Sentiment** to view the instant Negative badge and class probability distribution.
   - Type an unseen comment: *"The cabin crew was so warm and helpful! ❤️"*. Show instant Positive classification (`>85%` probability).
5. **Trends, Spikes & Model Evaluation (2:00 - 2:30):**
   - Navigate to **📈 Trends & Spikes** to highlight chronological volume trends and statistical spike alerts.
   - Navigate to **🎯 Model Performance** to show the test set confusion matrix and measured metrics (77.97% test accuracy, 0.7329 Macro F1).
6. **Executive Export (2:30 - 3:00):**
   - Navigate to **📑 Insights & Export**.
   - Review the auto-generated executive summary.
   - Click **Download Predictions CSV** and **Download Executive Report (.md)** to prove full end-to-end operational readiness.
