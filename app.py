import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)
st.title("❤️ Heart Disease Prediction System")

st.markdown("""
This application predicts the possibility of heart disease using a Machine Learning model.

Fill in the patient's information below and click **Predict**.
""")

st.sidebar.info("""
Heart Disease Prediction App

Machine Learning Model:
✔ Logistic Regression

Dataset:
UCI Heart Disease Dataset

Developed by:
Bhavya Sehgal
""")

st.markdown("""
<style>

.stButton>button{
    background-color:#ff4b4b;
    color:white;
    font-size:20px;
    border-radius:12px;
    height:55px;
}

.stNumberInput{
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

st.info("""
Model Used:
- Logistic Regression
- StandardScaler
- Machine Learning Project | Python • Streamlit • Scikit-Learn
- Accuracy: 82%
""")

# Load the trained model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")



# User Inputs
age = st.number_input("👤 Age",1,120,40)

sex = st.selectbox("Sex", ["Male", "Female"])

if sex == "Male":
    sex = 1
else:
    sex = 0

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    value=120
)

chol = st.number_input(
    "Cholesterol",
    value=200
)

fbs = st.selectbox("Fasting Blood Sugar", ["No", "Yes"])

if fbs == "Yes":
    fbs = 1
else:
    fbs = 0

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    value=150
)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

if exang == "Yes":
    exang = 1
else:
    exang = 0

oldpeak = st.number_input(
    "Oldpeak",
    value=1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)
if st.button("❤️Predict"):
     input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach,
                            exang, oldpeak,
                            slope, ca, thal]])
        

     input_scaled = scaler.transform(input_data)

     prediction = model.predict(input_scaled)

     if prediction[0] == 1:
      st.error("⚠️ High Risk of Heart Disease")
     else:
      st.success("✅ Low Risk of Heart Disease")