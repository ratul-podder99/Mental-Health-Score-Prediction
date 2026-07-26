# Mental Health Score Prediction

A machine learning web application that predicts a student's **mental health score (0–10)** based on their social media habits, academic life, and lifestyle patterns. The project covers the full pipeline — data exploration, preprocessing, model training and tuning, a FastAPI backend, and a vanilla JS/HTML/CSS frontend — and is deployed live via Render.

> ⚠️ **Disclaimer:** This tool is for informational and educational purposes only. It is not a clinical or diagnostic instrument. If you or someone you know is struggling, please reach out to a mental health professional or someone you trust.

---

## Live Link

https://mental-health-score-prediction-6-g06h.onrender.com

## Overview

Excessive or poorly managed social media use has been linked to changes in student well-being. This project uses a dataset of student habits — screen time, sleep, study hours, physical activity, stress level, and more — to train a regression model that estimates a **Mental Health Score** on a 0–10 scale, then serves that model through a REST API and an interactive web form.

## Features

- **Exploratory Data Analysis (EDA)** — distribution plots, correlation heatmaps, boxplots of stress vs. mental health score, and scatterplots of usage/sleep vs. score.
- **Data cleaning & preprocessing** — duplicate removal, outlier detection via IQR, skew correction, and grouping of low-frequency countries into an "Other" category.
- **Feature engineering pipeline** — built with `scikit-learn`'s `ColumnTransformer`, combining:
  - Standard scaling for numeric features
  - Log transformation + scaling for the skewed `Study_Hours` feature
  - Ordinal encoding for `Stress_Level` (Low → Medium → High → Very High)
  - One-hot encoding for categorical features (gender, academic level, platform, purpose of use, grouped country)
- **Model comparison** — Linear Regression vs. Random Forest Regressor, with hyperparameter tuning via `RandomizedSearchCV`.
- **REST API** — a FastAPI service exposing a `/predict` endpoint with request validation via Pydantic.
- **Interactive frontend** — a single-page form with real-time validation, a segmented stress-level selector, and an animated gauge that visualizes the predicted score.
- **Deployed backend** — the FastAPI service is deployed and live on Render.

## Dataset

- **File:** `Student Social Media And Mental Health Impact.csv`
- **Size:** 5,000 rows × 13 columns
- **Target variable:** `Mental_Health_Score`

| Column | Description |
|---|---|
| `Age` | Student's age |
| `Gender` | Male / Female |
| `Country` | Student's country (grouped into top 10 + "Other" for modeling) |
| `Academic_Level` | High School / Undergraduate / Graduate |
| `Most_Used_Platform` | Primary social media platform used |
| `Purpose_Of_Use` | Networking / Education / Entertainment / News |
| `Avg_Daily_Usage_Hours` | Average daily screen time (hours) |
| `Daily_Unlocks` | Number of phone unlocks per day |
| `Study_Hours` | Daily study hours |
| `Physical_Activity_Hours` | Daily physical activity (hours) |
| `Sleep_Hours_Per_Night` | Average nightly sleep (hours) |
| `Stress_Level` | Low / Medium / High / Very High |
| `Mental_Health_Score` | Target — self-reported mental health score (0–10) |

The dataset was clean (no missing values), and duplicates were dropped during preprocessing. Countries outside the top 10 by frequency were grouped into an `Other` category (`Grouped_country`) to reduce cardinality before one-hot encoding.

## Modeling Approach

Three models were trained and evaluated on a 70/30 train-test split (`random_state=42`):

| Model | Test R² | Train R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.740 | 0.724 | 0.536 | 0.676 |
| **Random Forest (default)** | **0.878** | 0.981 | **0.347** | **0.462** |
| Random Forest (tuned via RandomizedSearchCV) | 0.865 | 0.955 | 0.369 | 0.486 |

- Hyperparameter tuning (`RandomizedSearchCV`, 5-fold CV, 15 iterations) searched over `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- Interestingly, the **default Random Forest configuration outperformed the tuned version on the held-out test set**, so the default Random Forest pipeline (full preprocessing + `RandomForestRegressor`) was selected as the final model and serialized with `joblib` as `Mental_Health_Model.pkl`.

## Tech Stack

**Modeling / Data Science**
- Python, Pandas, NumPy
- scikit-learn (Pipelines, ColumnTransformer, RandomForestRegressor, RandomizedSearchCV)
- Matplotlib, Seaborn (EDA visualizations)
- Jupyter Notebook

**Backend**
- FastAPI
- Pydantic (request/response validation)
- Uvicorn (ASGI server)
- joblib (model serialization/loading)

**Frontend**
- HTML5, CSS3, vanilla JavaScript
- Fetch API for communicating with the backend

**Deployment**
- Render (backend API hosting)

## Project Structure

```
Mental-Health-Score-Prediction/
├── Mental_Health_Score.ipynb                       # EDA, preprocessing, model training & evaluation
├── Mental_Health_Model.pkl                         # Serialized final model pipeline (preprocessing + Random Forest)
├── Student Social Media And Mental Health Impact.csv  # Source dataset
├── main.py                                         # FastAPI application and /predict endpoint
├── requirements.txt                                # Backend dependencies
├── index.html                                      # Frontend markup
├── style.css                                       # Frontend styling
└── script.js                                       # Frontend logic (form handling, API calls, gauge animation)
```

## API Reference

### `GET /`
Health check endpoint — returns a simple welcome message.

### `POST /predict`
Accepts a JSON payload describing a student's habits and returns a predicted mental health score.

**Request body:**

```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5,
  "daily_unlocks": 80,
  "study_hours": 3.0,
  "physical_activity_hours": 1.5,
  "sleep_hours_per_night": 6.5,
  "stress_level": "High"
}
```

**Response body:**

```json
{
  "predicted_mental_health_score": 5.8
}
```

**Validation rules:**
- `age`: integer, 10–100
- `gender`: `"Male"` or `"Female"`
- `academic_level`: `"Undergraduate"`, `"Graduate"`, or `"High School"`
- `most_used_platform`: one of Facebook, LinkedIn, Instagram, Snapchat, Twitter, YouTube, TikTok, LINE, KakaoTalk, VKontakte, WhatsApp, WeChat
- `purpose_of_use`: `"Networking"`, `"Education"`, `"Entertainment"`, or `"News"`
- `avg_daily_usage_hours`, `study_hours`, `physical_activity_hours`, `sleep_hours_per_night`: floats, 0–24
- `daily_unlocks`: non-negative integer
- `stress_level`: `"Low"`, `"Medium"`, `"High"`, or `"Very High"`

Countries not among the model's top 10 most frequent training countries are automatically grouped into `"Other"` before prediction.

## Running Locally

**Backend:**

```bash
# Clone the repository
git clone <this-repository>
cd Mental-Health-Score-Prediction

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload
```

The API will be available locally, with interactive Swagger docs at `/docs`.

**Frontend:**

Simply open `index.html` in a browser (or serve it with any static file server). Update the `API_BASE` constant in `script.js` to point to your local or deployed backend URL.

## Deployment

The FastAPI backend is deployed on Render, running the same `main.py` application with `Mental_Health_Model.pkl` loaded at startup. CORS is enabled on the API to allow requests from the deployed frontend.

## Future Improvements

- Add cross-validation-based confidence intervals around predictions
- Expand the dataset with more diverse countries and platforms
- Add model explainability (e.g., SHAP values) to the frontend results
- Containerize the backend with Docker for more portable deployment
- Add automated tests for the API and preprocessing pipeline

## License

This project is intended for educational and portfolio purposes.