# app.py
import streamlit as st
import joblib
import numpy as np

# ==========================
# Load model and scaler
# ==========================
scaler = joblib.load("C:/python/machine_efficiency/models/scaler.pkl")
model = joblib.load("C:/python/machine_efficiency/models/random_forest_model.pkl")
le = joblib.load("C:/python/machine_efficiency/models/label_encoder.pkl")  # <- load LabelEncoder

# ==========================
# Helper function for energy savings
# ==========================
def calculate_energy_saved(failure_prob):
    """
    Estimate energy saved based on failure probability.
    """
    base_energy_loss = 100  # Assume 100 kWh loss if machine fails
    saved_energy = failure_prob * base_energy_loss
    return saved_energy

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Machine Efficiency & Clean Energy Monitor", layout="wide")
st.title("🌱 Machine Efficiency & Predictive Maintenance Dashboard")
st.write("Predict machine health and estimate energy savings if repaired proactively.")

st.subheader("🔧 Enter Machine Parameters")

# Dropdown with proper categories
machine_type = st.selectbox("Machine Type", le.classes_)

# Text inputs
air_temp = st.text_input("Air Temperature [K]", "300.0")
process_temp = st.text_input("Process Temperature [K]", "310.0")
rot_speed = st.text_input("Rotational Speed [rpm]", "1500")
torque = st.text_input("Torque [Nm]", "40.0")
tool_wear = st.text_input("Tool Wear [min]", "10.0")


# ==== Convert to float ====
try:
    air_temp = float(air_temp)
    process_temp = float(process_temp)
    rot_speed = float(rot_speed)
    torque = float(torque)
    tool_wear = float(tool_wear)
except ValueError:
    st.error("⚠️ Please enter valid numeric values.")
    st.stop()  # stop execution if conversion fails

# ==== Prepare features ====
type_encoded = le.transform([machine_type])[0]
features = np.array([[type_encoded, air_temp, process_temp, rot_speed, torque, tool_wear]])

# Scale numeric features only
features[:, 1:] = scaler.transform(features[:, 1:])

# ==== Prediction ====
if st.button("🔍 Predict Machine Health"):
    prediction = model.predict(features)[0]
    failure_prob = model.predict_proba(features)[0][1]
    energy_saved = failure_prob * 100

    # ===== Threshold-based warning =====
    threshold = 0.3  # warn if failure probability > 30%
    if failure_prob > threshold:
        st.warning(f"⚠️ Machine might fail soon! (Failure probability: {failure_prob:.2%})")
        if prediction == 1:
            st.error(f"⚠️ Machine needs repair! (Prediction = 1)")
        st.info(f"🌍 Estimated Energy Saved if repaired: **{energy_saved:.2f} kWh**")
    else:
        st.success(f"✅ Machine is operating normally. (Failure probability: {failure_prob:.2%})")