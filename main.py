from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path


app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model_Pipeline.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# MODEL COLUMNS
# ============================================================

COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood",
]


# ============================================================
# INPUT VALIDATION
# ============================================================

class Features(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180
    )

    price: float = Field(
        ...,
        gt=0
    )

    minimum_nights: int = Field(
        ...,
        ge=1,
        le=365
    )

    number_of_reviews: int = Field(
        ...,
        ge=0
    )

    reviews_per_month: float = Field(
        ...,
        ge=0
    )

    calculated_host_listings_count: int = Field(
        ...,
        ge=0
    )

    availability_365: int = Field(
        ...,
        ge=0,
        le=365
    )

    neighbourhood_group: str = Field(
        ...,
        min_length=1
    )

    neighbourhood: str = Field(
        ...,
        min_length=1
    )


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def greet():
    return {
        "message": "NYC Airbnb Room Type Prediction API is running"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(features: Features):

    row = pd.DataFrame(
        [features.model_dump()],
        columns=COLUMNS
    )

    prediction = model.predict(row)
    probability = model.predict_proba(row)

    return {
        "Predicted_room_type": prediction[0],
        "Probability": probability.tolist()[0]
    }