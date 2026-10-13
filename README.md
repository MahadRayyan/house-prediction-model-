# 🏠 California Housing Price Prediction API

A machine learning REST API built with FastAPI that predicts California house prices using a trained Random Forest Regressor. Supports both single predictions via JSON and batch predictions via CSV file upload.

---

### ⚡ Technologies

- Python
- Scikit-Learn (RandomForestRegressor)
- FastAPI
- Pandas
- Joblib

---

### 🚀 Features

- Trained on the California Housing dataset with **100 decision trees**
- Model achieves an **average error of ~$32,773** with strong R² score
- `POST /predict` — single house prediction via JSON input with field validation
- `POST /predict_file` — batch prediction via CSV upload, returns results as downloadable CSV
- `GET /health` — model status, feature list, and average error at a glance
- Input validation via Pydantic (latitude/longitude bounds, positive-only fields)
- Confidence range returned with every prediction (`predicted price ± $32,773`)

---

### 🧠 The Process

Started by exploring the California Housing dataset — checking for nulls, reviewing distributions, and understanding the 8 features. Then trained a Random Forest model with an 80/20 train-test split, evaluated it using MAE and R², and saved both the model and feature list using Joblib.

Built the FastAPI layer on top with three endpoints. The single-predict endpoint takes a JSON body, validates every field with Pydantic, runs it through the model, and returns the price in USD with a confidence range. The file endpoint handles CSV uploads, checks for missing columns, runs batch predictions, appends a new price column, and streams the result back as a downloadable CSV — so you can predict hundreds of houses at once without any extra setup.

---

### 🛠 Running the Project

1. Install dependencies:
   ```
   pip install fastapi uvicorn scikit-learn pandas joblib python-multipart
   ```
2. Train the model (generates `house_model.joblib` and `house_feature.joblib`):
   ```
   python train.py
   ```
3. Start the API:
   ```
   uvicorn main:app --reload
   ```
4. Open the interactive docs at `http://localhost:8000/docs`

---

### 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message and status |
| GET | `/health` | Model info and average error |
| POST | `/predict` | Single house price prediction |
| POST | `/predict_file` | Batch prediction from CSV upload |

**Example request body for `/predict`:**
```json
{
  "MedInc": 5.0,
  "HouseAge": 20.0,
  "AveRooms": 6.0,
  "AveBedrms": 1.2,
  "Population": 800.0,
  "AveOccup": 3.0,
  "Latitude": 34.05,
  "Longitude": -118.24
}
```

**Example response:**
```json
{
  "predicted price": "$245,000",
  "predicted_price_short": "$2.45 hundred thousand",
  "confidence_range": "$212,227 to $277,773"
}
```

---

### 📁 Project Structure

```
├── explore.py            # Dataset exploration and statistics
├── train.py              # Model training and evaluation
├── main.py               # FastAPI application
├── house_model.joblib    # Saved trained model
├── house_feature.joblib  # Saved feature names
└── test_houses.csv       # Sample CSV for batch prediction testing
```
