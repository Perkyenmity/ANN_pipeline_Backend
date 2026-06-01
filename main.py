from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json

# ── Initialize FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Risk Prediction API",
    description="ANN model that predicts whether a loan will be approved based on applicant details.",
    version="1.0.0"
)

# ── Load model and preprocessor once at startup ─────────────────────────────
print("Loading model and preprocessor...")

model = tf.keras.models.load_model("loan_ann_model.keras")
preprocessor = joblib.load("preprocessor.pkl")

with open("feature_columns.json", "r") as f:
    feature_cols = json.load(f)

NUMERIC_COLS     = feature_cols["numeric"]       # ['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsExperience']
CATEGORICAL_COLS = feature_cols["categorical"]   # ['Gender', 'Education', 'City', 'EmploymentType']

print("✅ Model loaded successfully!")
print(f"   Numeric columns    : {NUMERIC_COLS}")
print(f"   Categorical columns: {CATEGORICAL_COLS}")


# ── Input Schema ─────────────────────────────────────────────────────────────
# Literal types enforce valid values — the API will reject anything outside these lists

class LoanApplication(BaseModel):

    # --- Numeric fields ---
    Age: float = Field(..., gt=0, lt=120, example=35,
                       description="Applicant age in years")

    Income: float = Field(..., gt=0, example=75000,
                          description="Annual income in USD")

    LoanAmount: float = Field(..., gt=0, example=20000,
                              description="Requested loan amount in USD")

    CreditScore: float = Field(..., ge=300, le=850, example=720,
                               description="Credit score (300–850)")

    YearsExperience: float = Field(..., ge=0, example=8,
                                   description="Years of work experience")

    # --- Categorical fields (only valid values accepted) ---
    Gender: Literal["Female", "Male"] = Field(..., example="Male",
                                              description="Applicant gender")

    Education: Literal["High School", "Bachelors", "Masters", "PhD"] = Field(
        ..., example="Bachelors", description="Highest education level")

    City: Literal["Houston", "San Francisco", "New York", "Chicago"] = Field(
        ..., example="New York", description="City of residence")

    EmploymentType: Literal["Salaried", "Self-Employed", "Unemployed"] = Field(
        ..., example="Salaried", description="Employment type")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Income": 75000,
                "LoanAmount": 20000,
                "CreditScore": 720,
                "YearsExperience": 8,
                "Gender": "Male",
                "Education": "Bachelors",
                "City": "New York",
                "EmploymentType": "Salaried"
            }
        }


# ── Helper: determine risk level ─────────────────────────────────────────────
def get_risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "Low Risk"
    elif probability >= 0.50:
        return "Moderate Risk"
    elif probability >= 0.25:
        return "High Risk"
    else:
        return "Very High Risk"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message"         : "Loan Risk Prediction API is running!",
        "interactive_docs": "http://127.0.0.1:8000/docs",
        "predict_endpoint": "POST /predict"
    }


@app.get("/health")
def health_check():
    return {
        "status"      : "healthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type"          : "Artificial Neural Network (ANN)",
        "framework"           : "TensorFlow / Keras",
        "task"                : "Binary Classification — Loan Approval",
        "architecture"        : "Input(18) → Dense(32) → Dense(64) → Dense(128) → Dense(256) → Dense(512) → Dense(1, sigmoid)",
        "total_input_features": 18,
        "numeric_features"    : NUMERIC_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "valid_values": {
            "Gender"        : ["Female", "Male"],
            "Education"     : ["High School", "Bachelors", "Masters", "PhD"],
            "City"          : ["Houston", "San Francisco", "New York", "Chicago"],
            "EmploymentType": ["Salaried", "Self-Employed", "Unemployed"]
        }
    }


@app.post("/predict")
def predict_loan(application: LoanApplication):
    try:
        # Step 1 — Convert Pydantic model to dict, then DataFrame
        input_dict = application.dict()
        input_df   = pd.DataFrame([input_dict])

        # Step 2 — Reorder columns to exactly match training order
        # (numeric first, then categorical — same as your ColumnTransformer)
        input_df = input_df[NUMERIC_COLS + CATEGORICAL_COLS]

        # Step 3 — Apply the same ColumnTransformer used during training
        # StandardScaler on numeric + OneHotEncoder on categorical → 18 features
        processed_input = preprocessor.transform(input_df)

        # Step 4 — Run ANN prediction
        prediction_prob = model.predict(processed_input, verbose=0)
        probability     = float(prediction_prob[0][0])

        # Step 5 — Apply threshold (same as training: > 0.5 = Approved)
        approved = probability > 0.5

        return {
            "decision"              : "APPROVED" if approved else "REJECTED",
            "loan_approved"         : bool(approved),
            "approval_probability"  : round(probability, 4),
            "rejection_probability" : round(1 - probability, 4),
            "confidence"            : f"{round(probability * 100, 2)}%",
            "risk_level"            : get_risk_level(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

