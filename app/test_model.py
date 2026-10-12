import os
import joblib
import pandas as pd

# Get the project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the saved model
model_path = os.path.join(
    BASE_DIR,
    "models",
    "loan_approval_model.pkl"
)

# Load the complete trained pipeline
model = joblib.load(model_path)

# Example loan application
sample_application = pd.DataFrame([{
    "gender": "male",
    "married": "yes",
    "dependents": "0",
    "education": "graduate",
    "self_employed": "no",
    "applicantincome": 5000,
    "coapplicantincome": 1500,
    "loanamount": 150,
    "loan_amount_term": 360,
    "credit_history": 1,
    "property_area": "urban"
}])

# Make prediction
prediction = model.predict(sample_application)[0]
probability = model.predict_proba(sample_application)[0][1]

print("Model loaded successfully!")

if prediction == 1:
    print("Prediction: Approved")
else:
    print("Prediction: Not Approved")

print(f"Approval probability: {probability:.4f}")