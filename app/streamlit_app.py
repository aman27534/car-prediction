# pyrefly: ignore [missing-import]
import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction")
st.write("Enter your car details below and predict its selling price.")

st.subheader("Car Information")

col1, col2 = st.columns(2)

with col1:
    car_name = st.selectbox("Car Name", ["Maruti 800", "BMW", "XUV700", "Baleno", "Venue", "Mercedes", "i20", "Honda City"])
    company = st.selectbox("Company", ["Maruti", "Mahindra", "Hyundai", "Toyota", "Kia", "Mercedes Benz", "Audi", "BMW", "Honda", "Ford", "Tata", "Volkswagen"])
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

with col2:
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.number_input("Owner (0 = first owner)", min_value=0, max_value=3, value=0)
    kms_driven = st.number_input("Kms Driven", min_value=0, max_value=500000, value=50000, step=1000)

st.markdown("---")

if st.button("🔮 Predict Price", use_container_width=True):
    payload = {
        "Car_Name": car_name,
        "Company": company,
        "Year": year,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner,
        "Kms_Driven": kms_driven,
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"💰 Predicted Selling Price: **₹ {prediction:.2f} Lakhs**")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Make sure the FastAPI server is running on port 8000.")