from pathlib import Path
import csv
import pickle
import pandas as pd
import sqlite3
from datetime import datetime,timezone

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent


DB_PATH = BASE_DIR / "prediction_history.db"


def get_db_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                income REAL NOT NULL,
                house_age REAL NOT NULL,
                rooms REAL NOT NULL,
                bedrooms REAL NOT NULL,
                population REAL NOT NULL,
                model_name TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


initialize_database()


app = FastAPI(title="USA Housing Price Prediction")

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# Keep this order identical to the order used during training.
FEATURES = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]

# Map frontend choices to existing model files.
MODEL_FILES = {
    "LinearRegression": "LinearRegression.pkl",
    "Ridge": "Ridge.pkl",
    "Lasso": "Lasso.pkl",
    "ElasticNet": "ElasticNet.pkl",
    "RandomForestRegressor": "RandomForestRegressor.pkl",
    "HuberRegressor": "HuberRegressor.pkl",
    "KNeighborsRegressor": "KNeighborsRegressor.pkl",
    "SVR": "SVR.pkl",
    "SGDRegressor": "SGDRegressor.pkl",
    "PolynomialRegression": "PolynomialRegression.pkl",
    "MLPRegressor": "MLPRegressor.pkl",
    "LightGBM": "LightGBM.pkl",
    "XGBoost": "XGBoost.pkl",
}

# Load the models once when the application starts.
models = {}

for model_name, filename in MODEL_FILES.items():
    model_path = BASE_DIR / filename

    if model_path.exists():
        with open(model_path, "rb") as file:
            models[model_name] = pickle.load(file)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"models": list(models.keys())}
    )


@app.get("/results", response_class=HTMLResponse)
@app.get("/model", response_class=HTMLResponse)
async def model_page(request: Request):
    results_path = BASE_DIR / "model_results.csv"
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model results are missing. Run model.py to generate them."
        )

    with open(results_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)

    return templates.TemplateResponse(
        request=request,
        name="model.html",
        context={"columns": columns, "rows": rows}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    income: float = Form(...),
    house_age: float = Form(...),
    rooms: float = Form(...),
    bedrooms: float = Form(...),
    population: float = Form(...),
    model_name: str = Form(...),
):
    if model_name not in models:
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={"error": "Selected model is not available."},
            status_code=400
        )

    if not all(np.isfinite(v) for v in [
        income, house_age, rooms, bedrooms, population
    ]):
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={"error": "Please enter valid finite numbers."},
            status_code=400
        )

    input_data = np.array([[
        income,
        house_age,
        rooms,
        bedrooms,
        population,
    ]])
    model = models[model_name]
    prediction = float(model.predict(input_data)[0])

    # Save the successful prediction to SQLite
    created_at = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                income, house_age, rooms, bedrooms,
                population, model_name, predicted_price, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                income,
                house_age,
                rooms,
                bedrooms,
                population,
                model_name,
                prediction,
                created_at,
            ),
        )

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "prediction": round(prediction, 2),
            "model_name": model_name,
            "error": None,
        },
    )


@app.get("/api/models")
async def list_models():
    return {"available_models": list(models.keys())}


@app.get("/comparison", response_class=HTMLResponse)
async def comparison_page(request: Request):
    csv_path = BASE_DIR / "model_results.csv"

    if not csv_path.exists():
        return templates.TemplateResponse(
            request=request,
            name="comparison.html",
            context={
                "error": "model_results.csv was not found.",
                "columns": [],
                "rows": [],
                "charts": [],
                "best_model": None,
                "best_metric": None,
            },
        )

    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return templates.TemplateResponse(
            request=request,
            name="comparison.html",
            context={
                "error": "Could not read model_results.csv.",
                "columns": [],
                "rows": [],
                "charts": [],
                "best_model": None,
                "best_metric": None,
            },
        )

    if df.empty or len(df.columns) == 0:
        return templates.TemplateResponse(
            request=request,
            name="comparison.html",
            context={
                "error": "The results CSV is empty.",
                "columns": [],
                "rows": [],
                "charts": [],
                "best_model": None,
                "best_metric": None,
            },
        )

    # Clean up column names.
    df.columns = [str(col).strip() for col in df.columns]

    # Identify the model-name column.
    model_col = next(
        (
            col for col in df.columns
            if any(word in col.lower()
                   for word in ["model", "algorithm", "estimator"])
        ),
        None,
    )

    # If no recognizable name exists, use the first text column.
    if model_col is None:
        text_cols = df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        if text_cols:
            model_col = text_cols[0]

    # Find numeric columns that can be compared.
    metric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col != model_col
    ]

    # Build chart data using actual CSV values.
    if model_col is not None:
        model_labels = (
            df[model_col].fillna("Unknown").astype(str).tolist()
        )
    else:
        model_labels = [f"Model {i + 1}" for i in range(len(df))]

    charts = []

    for metric in metric_cols:
        values = [
            None if pd.isna(value) else float(value)
            for value in df[metric].tolist()
        ]

        charts.append({
            "metric": metric,
            "labels": model_labels,
            "values": values,
        })

    # Choose a suitable metric for the best-model indicator.
    best_model = None
    best_metric = None

    metric_names = {col.lower(): col for col in metric_cols}

    r2_col = next(
        (
            col for col in metric_cols
            if "r2" in col.lower().replace("^", "").replace("²", "2")
            or "r-squared" in col.lower()
        ),
        None,
    )

    error_col = next(
        (
            metric_names[name]
            for name in ["mae", "mean_absolute_error",
                         "rmse", "root_mean_squared_error", "mse"]
            if name in metric_names
        ),
        None,
    )

    # Prefer R² (higher is better), otherwise use an error metric.
    selected_metric = r2_col or error_col

    if selected_metric and model_col is not None:
        valid = df[selected_metric].notna()

        if valid.any():
            valid_values = df.loc[valid, selected_metric]

            if selected_metric == r2_col:
                best_index = valid_values.idxmax()
            else:
                best_index = valid_values.idxmin()

            best_model = str(df.loc[best_index, model_col])
            best_metric = {
                "name": selected_metric,
                "value": float(df.loc[best_index, selected_metric]),
                "rule": (
                    "Higher is better"
                    if selected_metric == r2_col
                    else "Lower is better"
                ),
            }

    # Convert CSV values into JSON-compatible table records.
    rows = __import__("json").loads(df.to_json(orient="records"))

    return templates.TemplateResponse(
        request=request,
        name="comparison.html",
        context={
            "error": None,
            "columns": df.columns.tolist(),
            "rows": rows,
            "charts": charts,
            "best_model": best_model,
            "best_metric": best_metric,
        },
    )

    return templates.TemplateResponse(
        request=request,
        name="comparison.html",
        context={"models": list(models.keys())}
    )
    
    
    
    
    
@app.get("/history", response_class=HTMLResponse)
async def prediction_history(request: Request):
    with get_db_connection() as connection:
        predictions = connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"predictions": predictions},
    )

