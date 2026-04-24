import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Load model and encoders
model = joblib.load("fertilizer_model.pkl")
le_soil = joblib.load("soil_encoder.pkl")
le_crop = joblib.load("crop_encoder.pkl")
le_fert = joblib.load("fert_encoder.pkl")

st.set_page_config(page_title="🌾 Fertilizer Dashboard", layout="centered")
st.title("🌱 Smart Fertilizer Recommendation System")
st.write("Enter your soil and crop details to get a fertilizer recommendation and nutrient dashboard!")

# Input columns
col1, col2 = st.columns(2)

with col1:
    temp = st.slider("Temperature (°C)", 0, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    soil = st.selectbox("Soil Type", le_soil.classes_)

with col2:
    crop = st.selectbox("Crop Type", le_crop.classes_)
    N = st.slider("Nitrogen (N)", 0, 100, 20)
    P = st.slider("Phosphorus (P)", 0, 100, 20)
    K = st.slider("Potassium (K)", 0, 100, 20)

# Predict button
if st.button("🌾 Predict Fertilizer"):
    data = pd.DataFrame([{
        "Temperature": temp,
        "Humidity": humidity,
        "Soil_Type": le_soil.transform([soil])[0],
        "Crop_Type": le_crop.transform([crop])[0],
        "N": N,
        "P": P,
        "K": K
    }])
    
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data).max()
    fertilizer_name = le_fert.inverse_transform([prediction])[0]
    
    st.success(f"🌱 Recommended Fertilizer: {fertilizer_name}")
    st.info(f"Confidence: {probability*100:.2f}%")
    
    # Nutrient Dashboard
    recommended_ranges = {
        "N": (30, 60),
        "P": (20, 40),
        "K": (20, 50)
    }
    
    nutrient_data = pd.DataFrame({
        "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
        "Your Value": [N, P, K],
        "Recommended Min": [recommended_ranges["N"][0], recommended_ranges["P"][0], recommended_ranges["K"][0]],
        "Recommended Max": [recommended_ranges["N"][1], recommended_ranges["P"][1], recommended_ranges["K"][1]]
    })
    
    st.subheader("📊 Nutrient Levels vs Recommended Ranges")
    
    # Prepare data for Altair
    chart_data = pd.melt(nutrient_data, id_vars="Nutrient", value_vars=["Your Value", "Recommended Min", "Recommended Max"],
                         var_name="Type", value_name="Value")
    
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("Nutrient:N"),
        y=alt.Y("Value:Q"),
        color=alt.Color("Type:N", scale=alt.Scale(range=["#1f77b4", "#2ca02c", "#d62728"])),
        tooltip=["Nutrient", "Type", "Value"]
    ).properties(width=600)
    
    st.altair_chart(chart)

# Footer / credit
st.markdown("---")
st.markdown("Designed by **ARMIYAU Nasiru Muhammad** 🌾")