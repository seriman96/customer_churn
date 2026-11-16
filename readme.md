# Telco Customer Churn Prediction — End-to-End MLE Project

An end-to-end machine learning project that predicts customer churn for a telecommunications company. This project covers the complete ML lifecycle, from data validation and model training to experiment tracking, API development, automated testing, containerization, CI/CD, and cloud deployment.

**Live API:** [Azure Container Apps](https://customer-churn-api.blueflower-5d1afa6f.canadacentral.azurecontainerapps.io)  
**API documentation:** [Swagger UI](https://customer-churn-api.blueflower-5d1afa6f.canadacentral.azurecontainerapps.io/docs)  
**Interactive application:** [Gradio UI](https://customer-churn-api.blueflower-5d1afa6f.canadacentral.azurecontainerapps.io/ui)

---

## Project Overview

Customer churn is an important business problem in the telecommunications industry. Identifying customers who are likely to leave allows businesses to understand retention risks and make informed decisions.

This project implements a binary classification solution using an XGBoost model to estimate the probability that a customer will churn.

Beyond model development, it demonstrates how to organize, test, package, deploy, and serve a machine learning application through a production-oriented workflow.

## Key Features

- **Data processing:** Data loading, preprocessing, feature engineering, and train/test splitting.
- **Data validation:** Great Expectations checks to validate input data.
- **Model training:** XGBoost binary classification.
- **Experiment tracking:** MLflow experiment tracking and model artifact management.
- **Model evaluation:** Precision, recall, F1-score, ROC AUC, and accuracy.
- **Model serving:** FastAPI prediction API.
- **Interactive interface:** Gradio application.
- **Automated testing:** Pytest-based test suite.
- **CI/CD:** GitHub Actions for automated testing and container publishing.
- **Container registry:** GitHub Container Registry (GHCR).
- **Cloud deployment:** Microsoft Azure Container Apps.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.12 |
| Data processing | Pandas, NumPy, SciPy |
| Machine learning | Scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Data validation | Great Expectations |
| Hyperparameter optimization | Optuna |
| API | FastAPI, Uvicorn, Pydantic |
| User interface | Gradio |
| Testing | Pytest, HTTPX |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Container registry | GitHub Container Registry |
| Cloud | Microsoft Azure Container Apps |
| Environment management | uv / Python virtual environments |

## Project Architecture

```text
                 CUSTOMER DATA
                       |
                       v
               DATA VALIDATION
              Great Expectations
                       |
                       v
              DATA PREPROCESSING
             & FEATURE ENGINEERING
                       |
                       v
                TRAIN / TEST
                    SPLIT
                       |
                       v
                MODEL TRAINING
                    XGBoost
                       |
                       v
               MODEL EVALUATION
                       |
                       v
               MLFLOW TRACKING
                       |
                       v
                MODEL PACKAGING
                       |
                       v
              FASTAPI + GRADIO
                       |
                       v
                 DOCKER IMAGE
                       |
                       v
              GITHUB CONTAINER
                   REGISTRY
                       |
                       v
             AZURE CONTAINER APPS
```

GitHub Actions automates testing and the container build-and-push workflow.

## Model Performance

The XGBoost model was evaluated on a held-out test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 0.720 |
| Precision | 0.484 |
| Recall | 0.834 |
| F1-score | 0.613 |
| ROC AUC | 0.837 |

The model achieved a ROC AUC of **0.837** and recall of **0.834**.

Recall is particularly relevant for churn prediction because identifying customers who are actually at risk of leaving is an important business objective.

The prediction API uses a configurable classification threshold of **0.35** (`CHURN_THRESHOLD`). This threshold can be adjusted depending on the business preference for balancing false positives and false negatives.

These results represent the evaluated test set and are not a guarantee of performance on future customer data.

## Data Validation

Great Expectations is integrated into the training pipeline to check the quality and consistency of input data before model training.

The latest verified pipeline execution successfully passed **23 out of 23 data validation checks**.

Great Expectations is included in the development dependencies rather than the production API requirements.

## Experiment Tracking and Model Management

MLflow is used to track model training experiments and manage model artifacts.

The training pipeline records the trained XGBoost model and its associated artifacts. The serving package includes the model files and supporting preprocessing metadata required for inference.

## API Endpoints

The FastAPI application exposes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Application and model health check |
| `/predict` | POST | Customer churn prediction |
| `/docs` | GET | Swagger API documentation |
| `/ui` | GET | Gradio interactive interface |

### Example health response

```json
{
  "status": "ok",
  "model": "loaded"
}
```

### Example prediction response

```json
{
  "prediction": "Likely to churn",
  "churn_probability": 0.6096,
  "threshold": 0.35
}
```

The probability represents the model's estimated likelihood of churn. The final classification is determined using the configured threshold.

For the complete request schema and an interactive prediction test, visit the [Swagger documentation](https://customer-churn-api.blueflower-5d1afa6f.canadacentral.azurecontainerapps.io/docs).

## Cloud Deployment

The application is deployed using **Microsoft Azure Container Apps**.

| Configuration | Value |
|---|---|
| Azure service | Container Apps |
| Resource group | `rg-customer-churn` |
| Container App | `customer-churn-api` |
| Region | Canada Central |
| Container registry | GitHub Container Registry |
| Container image | `ghcr.io/seriman96/customer_churn:latest` |
| Container port | 8000 |
| CPU | 0.5 vCPU |
| Memory | 1 GiB |
| Minimum replicas | 0 |
| Maximum replicas | 1 |
| Ingress | Public HTTPS |

The application uses scale-to-zero configuration to reduce resource consumption when there are no incoming requests.

**Deployment links:**

- [Live API](https://customer-churn-api.blueflower-5d1afa6f.canadacentral.azurecontainerapps.io)
- [Swagger documentation](https://customer-churn-api.blueflower-5d1afa6f.canadacentral.azurecontainerapps.io/docs)
- [Gradio interface](https://customer-churn-api.blueflower-5d1afa6f.canadacentral.azurecontainerapps.io/ui)

## CI/CD Pipeline

GitHub Actions automates the main integration and container delivery workflow.

The workflow contains three jobs:

1. **Test:** Installs development dependencies and executes the automated test suite.
2. **Build and Push:** Builds the Docker image and publishes it to GitHub Container Registry.
3. **Container Smoke Test:** Pulls the published image, starts the container, checks the health endpoint, and tests the prediction endpoint.

The container build and publishing workflow runs after successful tests on pushes to the main branch.

## Automated Testing

The project includes automated tests covering important application and machine learning components.

Latest verified test result:

```text
28 passed
```

The CI workflow also performs a container smoke test to verify that the packaged application starts and responds correctly.

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/seriman96/customer_churn.git
cd customer_churn
```

### 2. Create a virtual environment

Using Python 3.12:

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

Install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

The development requirements include the production dependencies.

### 4. Run automated tests

```bash
python -m pytest -v
```

### 5. Start the API

The packaged model must be available under `src/serving/model/`.

```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Open the following addresses locally:

- API: `http://127.0.0.1:8000`
- Health check: `http://127.0.0.1:8000/health`
- Swagger: `http://127.0.0.1:8000/docs`
- Gradio: `http://127.0.0.1:8000/ui`

## Running the Training Pipeline

The repository includes the Telco Customer Churn dataset.

Run the pipeline from the project root:

```bash
python scripts/run_pipeline.py --input WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn
```

The pipeline performs data validation, preprocessing, feature engineering, model training, evaluation, and MLflow logging.

## Additional Scripts

The `scripts/` directory contains utilities for the machine learning workflow:

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Main training pipeline |
| `run_model_tuning.py` | Model hyperparameter tuning |
| `prepare_processed_data.py` | Processed data preparation |
| `package_model.py` | Packages model artifacts for serving |
| `smoke_test_api.py` | API smoke testing |
| `run_pipeline_phase1_check.py` | Pipeline validation utility |

## Repository

**GitHub:** [seriman96/customer_churn](https://github.com/seriman96/customer_churn)

## Future Improvements

Potential extensions include:

- Model monitoring and prediction drift detection.
- Automated model retraining.
- Model explainability using SHAP.
- Improved model calibration and threshold optimization.
- Cloud-based experiment tracking and model management.
- Additional integration and load testing.

## Project Status

**Implemented and deployed:** End-to-end ML pipeline, data validation, XGBoost model, MLflow tracking, FastAPI, Gradio, automated tests, Docker, GitHub Actions, GHCR, and Azure Container Apps.

This project demonstrates practical machine learning engineering skills across model development, software engineering, testing, deployment, and cloud infrastructure.
