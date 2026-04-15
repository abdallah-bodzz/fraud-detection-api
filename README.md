# Fraud Detection API

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009485?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-7F52B3?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

[![AUPRC 0.87](https://img.shields.io/badge/AUPRC-0.87-006400?style=flat-square)](https://en.wikipedia.org/wiki/Precision-recall_curve)
[![Business-Driven Threshold](https://img.shields.io/badge/Business_Driven-Threshold-1E3A8A?style=flat-square)](README.md#results-on-held-out-test-set)
[![Production Ready](https://img.shields.io/badge/Production_Ready-FastAPI-FF4500?style=flat-square)](https://fastapi.tiangolo.com)

[![License: MIT](https://img.shields.io/badge/License-MIT-2E8B57?style=flat-square)](LICENSE)

I built this because most fraud detection demos stop at a Jupyter notebook with 89% accuracy — which is useless when 99.83% of transactions are legit. This is a production-ready API that makes decisions based on business cost, not academic metrics.

## What it actually does

Accepts a transaction's 30 features (Time, V1–V28, Amount), returns a fraud probability and a business-framed decision: block, review, or approve. The threshold is set to maximize dollars saved, not F1 score.

## Results (on held-out test set)

- AUPRC (the right metric for imbalanced data): **0.87**

- At threshold 0.4: Precision 87.3%, Recall 88.2%

- Net value protected over ~2 days of test data: **$10,576**

- Review costs incurred: $278

Why 0.4? A missed fraud costs ~$122 on average. A false alarm costs ~$2 to review. You lower the threshold until the cost of extra reviews outweighs the fraud you catch. Math over opinion.

## What's inside

```

fraud-detection-api/

├── src/

│   ├── main.py          # FastAPI app, routes, rate limiting, logging

│   ├── model.py         # Loads XGBoost, runs prediction, adds business note

│   ├── schemas.py       # Input/output validation (Pydantic)

│   ├── config.py        # Environment variables: threshold, paths, costs

│   └── utils.py         # Structured logging (loguru)

├── notebooks/

│   ├── 01_eda.ipynb

│   ├── 02_model_training.ipynb

│   └── 03_evaluation.ipynb   # Where threshold tuning happens

├── train_model.py       # Train once, saves model + scaler

├── Dockerfile

├── requirements.txt

└── .env.example

```

## How I run it

```bash

# Get data from Kaggle → data/creditcard.csv

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python train_model.py

uvicorn src.main:app --reload

```

Or with Docker:

```bash

docker build -t fraud-api .

docker run -p 8000:8000 fraud-api

```

## API

`POST /predict_transaction` — expects all 30 features, returns:

```json

{

  "fraud_probability": 0.0023,

  "is_fraud": false,

  "threshold_used": 0.4,

  "risk_level": "LOW",

  "business_note": "Transaction approved...",

  "transaction_amount": 149.62

}

```

`GET /health` — liveness check.

## Decisions I made (and why)

- **XGBoost over neural networks** – 30 numeric features, no missing values. Trees are faster, interpretable, and match NN performance on this scale.

- **scale_pos_weight over SMOTE** – SMOTE creates synthetic fraud samples by interpolating real ones. Those patterns might not exist in real fraud. Weighting the loss is cleaner.

- **AUPRC over ROC-AUC** – ROC looks good because of the huge TN pool. AUPRC measures what I actually care about: precision and recall on fraud.

- **Only Time and Amount scaled** – V1–V28 are PCA outputs, already normalized. Scaling them again would break that.

- **Rate limiting (60 req/min per IP)** – Simple in-memory sliding window. Good enough for a demo. In prod I'd use Redis.

## What I'd add if this went to prod

- Model versioning (MLflow)

- Online feature store for user history

- A/B testing for threshold changes

- Monitoring for data drift (fraud patterns change)

- More realistic review cost (I used $2, but real cost is higher)

## License

MIT (see LICENSE file)