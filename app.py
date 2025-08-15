import streamlit as st
import joblib
from streamlit_option_menu import option_menu
import numpy as np

# Load model and encoder
model = joblib.load("multi_model.pkl")
encoder = joblib.load("encoder.pkl")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Introduction", "Prediction", "Understanding Results"],
        icons=["house", "cloud-upload", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

st.title("Weather Prediction App")

# Introduction Page
if selected == "Introduction":
    st.title("🇿🇲 Zambia Weather & Power Impact Prediction App")
    st.markdown("""
    Welcome to the **Zambia Weather & Power Impact Prediction App** — your one-stop tool for predicting 
    **land surface temperature** and understanding how weather patterns may influence agriculture 
    and power availability.

    Zambia is a country blessed with rich agricultural land and abundant water resources, 
    but it also faces unique challenges. **Load shedding** — scheduled power outages due to electricity 
    supply shortages — can affect daily life, crop irrigation, and food production.

    This app combines **historical weather data**, **vegetation patterns**, and **precipitation trends** 
    to forecast land surface temperatures. By anticipating temperature shifts, farmers, businesses, 
    and households can better plan for:

    - 🌱 **Agriculture**: Predict planting & harvesting conditions.
    - 💡 **Load Shedding Impact**: Anticipate power outages during extreme weather.
    - 🌦 **Climate Awareness**: Track seasonal changes and prepare for weather extremes.

    Our goal?  
    **To help communities adapt, plan, and thrive — even in the face of climate and energy challenges.**
    """)

# Prediction Page
elif selected == "Prediction":
    st.header("Make a Weather Prediction")

    latitude = st.number_input("Latitude", format="%.6f")
    longitude = st.number_input("Longitude", format="%.6f")
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, step=1)
    quarter = st.number_input("Quarter (1-4)", min_value=1, max_value=4, step=1)
    province = st.selectbox("Province", encoder.classes_)

    lst_lag1 = st.number_input("Land Surface Temp (Lag 1)", format="%.2f")
    lst_lag2 = st.number_input("Land Surface Temp (Lag 2)", format="%.2f")
    lst_lag3 = st.number_input("Land Surface Temp (Lag 3)", format="%.2f")

    precip_lag1 = st.number_input("Precipitation (Lag 1)", format="%.2f")
    precip_lag2 = st.number_input("Precipitation (Lag 2)", format="%.2f")
    precip_lag3 = st.number_input("Precipitation (Lag 3)", format="%.2f")

    veg_lag1 = st.number_input("Vegetation Index (Lag 1)", format="%.2f")
    veg_lag2 = st.number_input("Vegetation Index (Lag 2)", format="%.2f")
    veg_lag3 = st.number_input("Vegetation Index (Lag 3)", format="%.2f")

    if st.button("Predict Weather"):
        province_enc = encoder.transform([province])[0]
        features = np.array([[latitude, longitude, month, quarter, province_enc,
                               lst_lag1, lst_lag2, lst_lag3,
                               precip_lag1, precip_lag2, precip_lag3,
                               veg_lag1, veg_lag2, veg_lag3]])
        prediction = model.predict(features)
        prediction_value = float(np.ravel(prediction)[0])
        st.success(f"Predicted Land Surface Temperature: {prediction_value:.2f} °C")

# Understanding Results Page
elif selected == "Understanding Results":
    st.header("How to Interpret the Results")
    st.write("""
        ### Variables Explained:
        - **Latitude & Longitude**: Exact location in Zambia.
        - **Month & Quarter**: Seasonal effects on temperature.
        - **Province**: Regional differences in weather patterns.
        - **Lag Features**: Past weather data used to predict future trends.

        ### How to Use Predictions:
        - **Load Shedding**: Anticipate higher demand during extreme temperatures.
        - **Agriculture**: Plan irrigation and planting schedules.
        - **Disaster Preparedness**: Be ready for heatwaves or unusual weather events.
    """)


