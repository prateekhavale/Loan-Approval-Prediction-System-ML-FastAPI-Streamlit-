from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import pickle
import pandas as pd

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# api Object
app = FastAPI()


# Input value Varification
class LoanInput(BaseModel):

    ApplicantIncome: Annotated[
        float,
        Field(..., gt=0, description="Monthly income of the loan applicant")
    ]

    CoapplicantIncome: Annotated[
        float,
        Field(..., ge=0, description="Monthly income of the co-applicant (0 if none)")
    ]

    LoanAmount: Annotated[
        float,
        Field(..., gt=0, description="Requested loan amount in thousands")
    ]

    Credit_History: Annotated[
        int,
        Field(..., ge=0, le=1, description="Credit history of applicant (1 = good, 0 = bad)")
    ]

    Loan_Amount_Term: Annotated[
        int,
        Field(..., gt=0, description="Loan repayment term in months")
    ]

    Gender: Annotated[
        Literal["Male", "Female"],
        Field(..., description="Gender of the applicant")
    ]

    Married: Annotated[
        Literal["Yes", "No"],
        Field(..., description="Marital status of the applicant")
    ]

    Dependents: Annotated[
        Literal["0", "1", "2", "3+"],
        Field(..., description="Number of dependents")
    ]

    Education: Annotated[
        Literal["Graduate", "Not Graduate"],
        Field(..., description="Education level of the applicant")
    ]

    Self_Employed: Annotated[
        Literal["Yes", "No"],
        Field(..., description="Whether the applicant is self-employed")
    ]

    Property_Area: Annotated[
        Literal["Urban", "Semiurban", "Rural"],
        Field(..., description="Location of the property")
    ]





@app.get("/")
def home():
    return {"message": "Loan Prediction API running"}


@app.post("/predict")
def predict(data: LoanInput):

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([{
            "ApplicantIncome": data.ApplicantIncome,
            "CoapplicantIncome": data.CoapplicantIncome,
            "LoanAmount": data.LoanAmount,
            "Credit_History": data.Credit_History,
            "Loan_Amount_Term": data.Loan_Amount_Term,
            "Gender": data.Gender,
            "Married": data.Married,
            "Dependents": data.Dependents,
            "Education": data.Education,
            "Self_Employed": data.Self_Employed,
            "Property_Area": data.Property_Area
        }])

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        label_map = {
            0:"Loan Not Approved",
            1:"Loan Approved"
        }

    

        return {
            "loan_approval": label_map[prediction],
            "probability": float(probability)
        }

    except Exception as e:
        return {"error": str(e)}