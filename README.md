# TRIP-FARE
# 🚖 TripFare: Predicting Urban Taxi Fare with Machine Learning

TripFare is a Streamlit web application that predicts New York City taxi fares based on ride features using machine learning. It also includes interactive visualizations to explore key trends in the dataset.

---

## 📌 Features

- ✅ Predict taxi fares using an XGBoost regression model  
- ✅ Input ride details like distance, duration, vendor, pickup hour, etc.  
- ✅ Explore ride data with interactive charts powered by Plotly  
- ✅ Visual analysis includes:
  - Fare vs Trip Distance
  - Trip Duration & Distance Distributions
  - Fare by Passenger Count
  - Trip Counts by Hour & Day of Week
  - Fare/km and Fare/min efficiency
  - Correlation Heatmap

---

## 📂 Dataset Overview

The model was trained on NYC Yellow Taxi data and includes the following key fields:

- `pickup_datetime`, `dropoff_datetime`  
- `pickup_longitude`, `pickup_latitude`  
- `dropoff_longitude`, `dropoff_latitude`  
- `passenger_count`, `total_amount`, `RatecodeID`, `VendorID`

Feature engineering was applied to compute:

- Trip Distance (via Haversine formula)  
- Trip Duration (in minutes)  
- Pickup Hour, AM/PM, Day of Week  
- Weekend / Night Indicators  
- Fare per km & per minute

---

## 📈 Model Performance

| Model                  | RMSE   | R² Score |
|------------------------|--------|----------|
| Linear Regression      | 0.358  | 0.872    |
| Decision Tree Regressor| 0.405  | 0.836    |
| Random Forest Regressor| 0.308  | 0.905    |
| **XGBoost Regressor**  | ✅ **0.297**  | ✅ **0.912**    |

---

## 🚀 Run the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/tripfare-prediction.git
cd tripfare-prediction
