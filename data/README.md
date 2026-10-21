# Dataset Documentation

## Primary Dataset: Twitter US Airline Sentiment

### 1. Dataset Origin & Source
- **Origin:** Crowdflower's *Data for Everyone* collection.
- **Hosted on:** Kaggle & Open Academic Repositories ([Crowdflower / Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)).
- **Timeframe:** Data collected in February 2015.
- **License / Terms of Use:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Permitted for research, educational, and non-commercial development.

### 2. Actual Dataset Statistics (Raw Data)
- **Total Records:** 14,640 rows, 15 columns.
- **Sentiment Classes & Distribution:**
  - `negative`: 9,178 (62.69%)
  - `neutral`: 3,099 (21.17%)
  - `positive`: 2,363 (16.14%)
- **Products / Entities Analyzed:**
  - United: 3,822
  - US Airways: 2,913
  - American: 2,759
  - Southwest: 2,420
  - Delta: 2,222
  - Virgin America: 504

### 3. Key Columns & Semantic Mapping
| Original Column Name | Framework Standard Field | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `text` | `text` | string | Raw customer tweet/comment content |
| `airline_sentiment` | `sentiment` | string | Sentiment category (`positive`, `negative`, `neutral`) |
| `tweet_created` | `date` | datetime/string | Timestamp of customer feedback |
| `airline` | `product` | string | Airline / brand being referenced |
| `negativereason` | `topic` | string | Categorized complaint issue (e.g., Late Flight, Customer Service) |
| `tweet_id` | `user_id` | numeric/string | Anonymous identifier (PII sanitized) |

### 4. Sample Dataset (`data/sample/sample_airline_sentiment.csv`)
- **Total Records:** 300 rows (100 Positive, 100 Negative, 100 Neutral).
- **Purpose:** Used for lightweight unit testing, local development, and fast CI validation without requiring the full raw file.
