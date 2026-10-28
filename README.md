# USA Housing Price Prediction

## Overview
A machine learning web application that predicts house prices using the USA Housing dataset. Built with Python and FastAPI.

## Features
- Predict house prices using trained machine learning models
- Compare the performance of different regression models
- View model evaluation results
- Store and view prediction history using SQLite
- User-friendly web interface with HTML templates

## Technologies Used
- Python
- FastAPI
- Scikit-learn
- Pandas and NumPy
- SQLite
- HTML and CSS
- Pickle

## Machine Learning Models
The project includes trained regression models such as:
- Linear Regression
- Random Forest Regressor
- Ridge Regression
- Lasso Regression
- ElasticNet
- K-Nearest Neighbors
- Support Vector Regression (SVR)
- Other regression models

## Project Structure
- `main.py` — FastAPI application
- `model.py` — Model-related code
- `model_results.csv` — Model evaluation results
- `USA_Housing - USA_Housing.csv` — Housing dataset
- `templates/` — HTML templates for the web interface
- `*.pkl` — Saved machine learning models

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/likithm19/ml-home-prediction.git
   ```

2. Open the project folder:

   ```bash
   cd ml-home-prediction
   ```

3. Install the required dependencies:

   ```bash
   pip install fastapi uvicorn pandas numpy scikit-learn
   ```

   Install any additional libraries required by the project code.

4. Start the application:

   ```bash
   uvicorn main:app --reload
   ```

5. Open `http://127.0.0.1:8000` in your browser.

## Prediction History
Prediction history is stored locally in an SQLite database created by the application.

## Author
Likith M
