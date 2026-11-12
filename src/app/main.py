from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.inference import predict, load_model


# Gradio is optional for local API-only use.
try:
    import gradio as gr
except ImportError:
    gr = None


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Predict customer churn using a trained XGBoost classification model."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------

class CustomerData(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No", "Yes", "No phone service"]

    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "message": "Telco Customer Churn Prediction API",
        "status": "ok",
        "docs": "/docs",
    }

'''
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_directory": "configured",
    }
'''

@app.get("/health")
def health():
    load_model()

    return {
        "status": "ok",
        "model": "loaded",
    }

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        return predict(customer.model_dump())

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the application logs.",
        ) from exc


# ---------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------

def gradio_predict(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    return predict(customer)


if gr is not None:
    demo = gr.Interface(
        fn=gradio_predict,
        inputs=[
            gr.Dropdown(["Female", "Male"], label="Gender", value="Female"),
            gr.Radio([0, 1], label="Senior Citizen", value=0),
            gr.Dropdown(["Yes", "No"], label="Partner", value="Yes"),
            gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
            gr.Number(label="Tenure (months)", value=12, precision=0),
            gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
            gr.Dropdown(
                ["No", "Yes", "No phone service"],
                label="Multiple Lines",
                value="No",
            ),
            gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet Service",
                value="DSL",
            ),
            gr.Dropdown(
                ["No", "Yes", "No internet service"],
                label="Online Security",
                value="No",
            ),
            gr.Dropdown(
                ["No", "Yes", "No internet service"],
                label="Online Backup",
                value="No",
            ),
            gr.Dropdown(
                ["No", "Yes", "No internet service"],
                label="Device Protection",
                value="No",
            ),
            gr.Dropdown(
                ["No", "Yes", "No internet service"],
                label="Tech Support",
                value="No",
            ),
            gr.Dropdown(
                ["No", "Yes", "No internet service"],
                label="Streaming TV",
                value="No",
            ),
            gr.Dropdown(
                ["No", "Yes", "No internet service"],
                label="Streaming Movies",
                value="No",
            ),
            gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract",
                value="Month-to-month",
            ),
            gr.Dropdown(
                ["Yes", "No"],
                label="Paperless Billing",
                value="Yes",
            ),
            gr.Dropdown(
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
                label="Payment Method",
                value="Electronic check",
            ),
            gr.Number(label="Monthly Charges", value=70.0),
            gr.Number(label="Total Charges", value=840.0),
        ],
        outputs=gr.JSON(label="Churn Prediction"),
        title="Telco Customer Churn Prediction",
        description=(
            "Enter customer information to estimate churn probability. "
            "The model uses a recall-oriented decision threshold."
        ),
    )

    app = gr.mount_gradio_app(
        app,
        demo,
        path="/ui",
    )