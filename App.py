import pickle
import streamlit as st
import numpy as np
import pandas as pd
import os

# ================= CHECK FILE =================
st.write("📂 Files in this folder:", os.listdir())

# ================= MODEL LOAD =================
model = None
model_path = 'rf_model.pkl'

if os.path.exists(model_path):
    try:
        model = pickle.load(open(model_path, 'rb'))
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error("❌ rf_model.pkl NOT FOUND in this folder!")

    st.write("📌 Model expects:", model.feature_names_in_)

# ================= TITLE =================
st.title('❤️ Heart attack risk classification app')
st.write("🚀 App is running...")

# ================= INPUT =================
Age = st.number_input('Age', min_value=20, max_value=100, value=25)
RestingBP = st.number_input('RestingBP', min_value=0, max_value=300, value=100)
Cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=100)
FastingBS = st.selectbox('FastingBS', (0, 1))
MaxHR = st.number_input('MaxHR', min_value=60, max_value=220, value=150)
Oldpeak = st.number_input('Oldpeak', min_value=-3.0, max_value=10.0, value=2.0)

gender = st.selectbox('Gender', ('M', 'F'))
ChestPainType_input = st.selectbox('ChestPainType', ('ATA', 'NAP', 'ASY', 'TA'))
RestingECG_input = st.selectbox('RestingECG', ('Normal', 'ST', 'LVH'))
ExerciseAngina = st.selectbox('ExerciseAngina', ('N', 'Y'))
ST_slope_input = st.selectbox('ST_slope', ('UP', 'Flat', 'Down'))

# ================= ENCODING =================
Excercise_Angina = 1 if ExerciseAngina == 'Y' else 0
Sex_F = 1 if gender == 'F' else 0
Sex_M = 1 if gender == 'M' else 0

ChestPainType = {'ATA':1,'NAP':2,'ASY':3,'TA':4}[ChestPainType_input]
RestingECG = {'Normal':1,'ST':2,'LVH':3}[RestingECG_input]
ST_slope = {'UP':1,'Flat':2,'Down':3}[ST_slope_input]

# ================= DATAFRAME =================
# ================= DATAFRAME (FIXED) =================

# Create with your current names first
input_df = pd.DataFrame({
    'Age':[Age],
    'RestingBP':[RestingBP],
    'Cholesterol':[Cholesterol],
    'FastingBS':[FastingBS],
    'MaxHR':[MaxHR],
    'Oldpeak':[Oldpeak],
    'Sex_M':[Sex_M],
    'Chest_PainType':[ChestPainType],
    'RestingECG':[RestingECG],
    'ExerciseAngina':[Excercise_Angina],
    'ST_Slope':[ST_slope]
})

# 🔥 IMPORTANT LINE (AUTO FIX)
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

st.write("📊 Final Input (matched to model):", input_df)

# ================= PREDICTION =================
if st.button('🔍 Predict'):

    if model is None:
        st.error("❌ Model not loaded. Put rf_model.pkl in this folder!")
    else:
        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.error('🚨 High Risk of Heart Attack 💔')
        else:
            st.success('😊 Low Risk of Heart Attack 💚')

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)
            st.write(f"📊 Risk Probability: {prob[0][1]*100:.2f}%")