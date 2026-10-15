"""Serve the churn model over HTTP.

Endpoints
    GET  /            HTML form for scoring one customer
    POST /predict     JSON in, churn probability out
    GET  /health      liveness check used by the Docker HEALTHCHECK
    GET  /model-info  training metrics written by train.py
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, get_args, get_origin

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "artifacts")))
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float
    model: str


# Form layout for the HTML page: (group title, [(field, label, default)]).
FORM_GROUPS = [
    (
        "Customer",
        [
            ("gender", "Gender", "Female"),
            ("SeniorCitizen", "Senior citizen", "No"),
            ("Partner", "Has a partner", "No"),
            ("Dependents", "Has dependents", "No"),
        ],
    ),
    (
        "Account and billing",
        [
            ("tenure", "Tenure (months)", 2),
            ("Contract", "Contract", "Month-to-month"),
            ("PaymentMethod", "Payment method", "Electronic check"),
            ("PaperlessBilling", "Paperless billing", "Yes"),
            ("MonthlyCharges", "Monthly charges ($)", 85.0),
            ("TotalCharges", "Total charges to date ($)", 170.0),
        ],
    ),
    (
        "Services",
        [
            ("PhoneService", "Phone service", "Yes"),
            ("MultipleLines", "Multiple lines", "No"),
            ("InternetService", "Internet service", "Fiber optic"),
            ("OnlineSecurity", "Online security", "No"),
            ("OnlineBackup", "Online backup", "No"),
            ("DeviceProtection", "Device protection", "No"),
            ("TechSupport", "Tech support", "No"),
            ("StreamingTV", "Streaming TV", "Yes"),
            ("StreamingMovies", "Streaming movies", "Yes"),
        ],
    ),
]


def build_form_spec() -> list[dict]:
    """Derive form controls from the Customer schema so the two never drift."""
    groups = []
    for title, fields in FORM_GROUPS:
        controls = []
        for name, label, default in fields:
            annotation = Customer.model_fields[name].annotation
            if get_origin(annotation) is Literal:
                controls.append(
                    {"name": name, "label": label, "kind": "select",
                     "options": list(get_args(annotation)), "default": default}
                )
            else:
                step = "1" if annotation is int else "0.01"
                controls.append(
                    {"name": name, "label": label, "kind": "number",
                     "step": step, "default": default}
                )
        groups.append({"title": title, "controls": controls})
    return groups


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast: if the model is missing the container exits instead of
    # starting a server that can only return errors.
    model_path = MODEL_DIR / "model.joblib"
    metrics_path = MODEL_DIR / "metrics.json"
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Run train.py first.")
    app.state.model = joblib.load(model_path)
    app.state.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    yield


app = FastAPI(title="Telco Churn API", version="1.0.0", lifespan=lifespan)
app.mount("/statics", StaticFiles(directory=BASE_DIR / "statics"), name="statics")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"groups": build_form_spec(), "metrics": app.state.metrics, "threshold": THRESHOLD},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.get("/model-info")
def model_info():
    return {"threshold": THRESHOLD, **app.state.metrics}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    # Reorder columns to match training so scikit-learn's feature-name check passes.
    frame = pd.DataFrame([customer.model_dump()])[app.state.model.feature_names_in_]
    probability = float(app.state.model.predict_proba(frame)[0, 1])
    return Prediction(
        churn_probability=round(probability, 4),
        will_churn=probability >= THRESHOLD,
        threshold=THRESHOLD,
        model=app.state.metrics.get("selected_model", "unknown"),
    )
