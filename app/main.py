
import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "loan_approval_model.pkl"
)

# Load the trained machine-learning pipeline
model = joblib.load(MODEL_PATH)
CANDIDATE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "loan_approval_candidate.pkl"
)
candidate_model = joblib.load(CANDIDATE_MODEL_PATH)
# Create FastAPI application
app = FastAPI(title="LoanWise Prediction API")

# Allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input data from the frontend
class LoanApplication(BaseModel):
    gender: str
    married: str
    dependents: str
    education: str
    self_employed: str

    applicantincome: float = Field(ge=0)
    coapplicantincome: float = Field(ge=0)
    loanamount: float = Field(gt=0)
    loan_amount_term: float = Field(gt=0)
    credit_history: float

    property_area: str


@app.get("/")
def home():
    return {
        "message": "LoanWise prediction API is running successfully!"
    }


@app.post("/predict")
def predict_loan(application: LoanApplication):
    try:
        # Convert the loan amount from rupees to the
        # thousands-based scale used in the training dataset.
        loan_amount_in_dataset_units = application.loanamount / 1000

        # Keep the feature names and order consistent
        # with the trained model's input data.
        input_data = pd.DataFrame([{
            "gender": application.gender,
            "married": application.married,
            "dependents": application.dependents,
            "education": application.education,
            "self_employed": application.self_employed,
            "applicantincome": application.applicantincome,
            "coapplicantincome": application.coapplicantincome,
            "loanamount": loan_amount_in_dataset_units,
            "loan_amount_term": application.loan_amount_term,
            "credit_history": application.credit_history,
            "property_area": application.property_area,
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # This follows the existing model test's convention:
        # class 1 represents loan approval.
        approval_probability = float(probabilities[1])

        result = "Approved" if prediction == 1 else "Not Approved"

        return {
            "prediction": result,
            "approval_probability": approval_probability,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        ) from error


@app.post("/predict-candidate")
def predict_candidate(application: LoanApplication):
    try:
        loan_amount_in_dataset_units = application.loanamount / 1000
        input_data = pd.DataFrame([{
            "gender": application.gender,
            "married": application.married,
            "dependents": application.dependents,
            "education": application.education,
            "self_employed": application.self_employed,
            "applicantincome": application.applicantincome,
            "coapplicantincome": application.coapplicantincome,
            "loanamount": loan_amount_in_dataset_units,
            "loan_amount_term": application.loan_amount_term,
            "credit_history": application.credit_history,
            "property_area": application.property_area,
        }])
        prediction = candidate_model.predict(input_data)[0]
        probabilities = candidate_model.predict_proba(input_data)[0]
        return {
            "prediction": "Approved" if prediction == 1 else "Not Approved",
            "approval_probability": float(probabilities[1]),
            "model": "Logistic Regression Candidate",
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Candidate prediction failed: {str(error)}",
        ) from error
