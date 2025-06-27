import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import pytz

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="TripFare Dashboard", layout="wide")
st.title("🚖 TripFare: Urban Taxi Fare Estimator")

# ---------------------- Load Model ----------------------
model_path = r"C:\Users\Rakshitha\TRIP PROJECT\xgb_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ---------------------- Haversine Function ----------------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))  # in km

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Rakshitha\Downloads\taxi_fare.csv")
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
    df['trip_distance'] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'],
                                                          row['dropoff_longitude'], row['dropoff_latitude']), axis=1)
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.day_name()
    df['is_weekend'] = df['pickup_datetime'].dt.dayofweek >= 5
    df['is_night'] = df['pickup_hour'].apply(lambda x: x < 6 or x > 22)
    df['am_pm'] = df['pickup_datetime'].dt.strftime('%p')
    df['fare_per_km'] = df['total_amount'] / (df['trip_distance'] + 0.01)
    df['fare_per_min'] = df['total_amount'] / (df['trip_duration'] + 0.01)
    df = df[(df['trip_distance'] > 0) & (df['trip_duration'] > 0) & (df['total_amount'] > 0)]
    return df

df = load_data()

# ---------------------- Tabs ----------------------
tab1, tab2, tab3 = st.tabs(["📊 Predict Fare", "📈 Visual Analysis", "ℹ️ Project Info"])

# ---------------------- Tab 1: Predict Fare ----------------------
with tab1:
    with st.form("fare_form"):
        st.header("🧾 Enter Ride Details")

        col1, col2 = st.columns(2)
        with col1:
            passenger_count = st.number_input("Passenger Count", 1, 6, value=1)
            trip_distance = st.number_input("Trip Distance (km)", min_value=0.1, value=2.5)
            trip_duration = st.number_input("Trip Duration (minutes)", min_value=1.0, value=10.0)
            vendor_id = st.selectbox("Vendor ID", [1, 2])
            ratecode = st.selectbox("Ratecode ID", [1, 2])
        with col2:
            pickup_hour = st.slider("Pickup Hour (0–23)", 0, 23, 12)
            is_weekend = st.radio("Weekend Ride?", ["No", "Yes"])
            is_night = st.radio("Night Ride?", ["No", "Yes"])
            am_pm = st.radio("Time Period", ["AM", "PM"])

        submitted = st.form_submit_button("🚀 Predict Fare")

    if submitted:
        input_array = np.array([[np.log1p(trip_distance),
                                 np.log1p(trip_duration),
                                 passenger_count,
                                 pickup_hour,
                                 1 if is_weekend == "Yes" else 0,
                                 1 if is_night == "Yes" else 0,
                                 1 if am_pm == "PM" else 0,
                                 1 if vendor_id == 2 else 0,
                                 1 if ratecode == 2 else 0]])

        pred_log = model.predict(input_array)[0]
        pred_fare = np.expm1(pred_log)

        st.success(f"💵 Predicted Total Fare: **${pred_fare:.2f}**")

        st.markdown("### 🚖 Ride Summary")
        st.markdown(f"""
        - **Passenger Count:** {passenger_count}  
        - **Distance:** {trip_distance:.2f} km  
        - **Duration:** {trip_duration:.1f} min  
        - **Vendor ID:** {vendor_id}  
        - **Ratecode ID:** {ratecode}  
        - **Hour:** {pickup_hour}  
        - **AM/PM:** {am_pm}  
        - **Weekend:** {'Yes' if is_weekend == 'Yes' else 'No'}  
        - **Night Ride:** {'Yes' if is_night == 'Yes' else 'No'}  
        """)

# ---------------------- Tab 2: Visual Analysis ----------------------
import plotly.express as px
import plotly.graph_objects as go

# ---------------------- Tab 2: Visual Analysis ----------------------
with tab2:
    st.subheader("📊 Data Visualizations and Insights")

    st.markdown("### 💰 Fare vs Trip Distance")
    st.markdown("""
    Fare increases with distance. Most rides fall under 10 km. Outliers may include tolls or long waiting times.
    """)
    fig1 = px.scatter(df.sample(1000), x='trip_distance', y='total_amount',
                      title="Fare vs Trip Distance",
                      template="plotly_dark", opacity=0.6,
                      labels={'trip_distance': 'Distance (km)', 'total_amount': 'Total Fare ($)'})
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### ⏱️ Trip Duration Distribution")
    st.markdown("""
    Most taxi rides last between 5–30 minutes. Few rides exceed 60 minutes.
    """)
    fig2 = px.histogram(df, x="trip_duration", nbins=40, title="Trip Duration Histogram",
                        template="plotly_dark", color_discrete_sequence=['#00cc96'])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 👥 Fare by Passenger Count")
    st.markdown("""
    Fare does not vary much with passenger count. Taxi fare is mostly fixed per ride.
    """)
    fig3 = px.box(df, x='passenger_count', y='total_amount',
                  title="Passenger Count vs Fare",
                  template="plotly_dark", color_discrete_sequence=['#636efa'])
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 📏 Trip Distance Distribution")
    st.markdown("""
    Most trips are within 1 to 5 km. Longer trips are rare but significantly impact fare.
    """)
    fig4 = px.histogram(df, x="trip_distance", nbins=40,
                        title="Trip Distance Histogram",
                        template="plotly_dark", color_discrete_sequence=['#ab63fa'])
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 🕒 Trip Count by Hour")
    st.markdown("""
    Demand peaks during commute hours (8–9 AM and 5–7 PM). Few rides occur late at night.
    """)
    fig5 = px.histogram(df, x="pickup_hour", title="Trip Count by Hour",
                        template="plotly_dark", color_discrete_sequence=['#EF553B'])
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### 📆 Trip Count by Day of Week")
    st.markdown("""
    Weekday trips are more frequent than weekends, showing strong work-related usage.
    """)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['pickup_day'] = pd.Categorical(df['pickup_day'], categories=days_order, ordered=True)
    fig6 = px.histogram(df.sort_values('pickup_day'), x="pickup_day",
                        title="Trip Count by Day",
                        template="plotly_dark", color_discrete_sequence=['#FFA15A'])
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("### 💸 Fare per KM and Minute")
    st.markdown("""
    Most fares are below $5/km and $2/min. High values indicate heavy traffic or tolls.
    """)
    fig7 = go.Figure()
    fig7.add_trace(go.Histogram(x=df['fare_per_km'], name='Fare/km', opacity=0.6, marker_color='lightgreen'))
    fig7.add_trace(go.Histogram(x=df['fare_per_min'], name='Fare/min', opacity=0.6, marker_color='orange'))
    fig7.update_layout(
        barmode='overlay',
        title="Fare Distribution per KM and per Minute",
        template='plotly_dark',
        xaxis_title="Fare Value",
        yaxis_title="Count"
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("### 🔍 Correlation Heatmap")
    st.markdown("""
    Fare is strongly correlated with trip distance and duration. These are the top features used in modeling.
    """)
    corr_matrix = df.select_dtypes(include='number').corr().round(2)
    fig8 = px.imshow(corr_matrix,
                     text_auto=True,
                     color_continuous_scale='RdBu',
                     title="Feature Correlation Heatmap",
                     template="plotly_dark")
    st.plotly_chart(fig8, use_container_width=True)
