import streamlit as st
import pandas as pd
import numpy as np
from helper import preprocess
import pickle

# Load the model
model = pickle.load(open('saved_models/decision_tree.sav', 'rb'))
scaler = pickle.load(open('saved_models/standard_scaler.sav', 'rb'))

# Define the Streamlit app
st.title('Cardiovascular Disease Prediction')

feature_list = dict()

# Add input widgets for user input
feature_list['gender'] = st.selectbox(label='Gender', options=['Female', 'Male'])
feature_list['age'] = st.number_input('Enter Age:', format='%.0f')
feature_list['weight'] = st.number_input('Enter weight (kg):', format='%.0f')
feature_list['height'] = st.number_input('Enter height (cm):', format='%.0f')
feature_list['active'] = st.selectbox(label='Physical Activity (Active: routine weekly exercise)', options=['Active', 'Normal'])
feature_list['alco'] = st.selectbox(label='Consume Alcohol', options=['Yes', 'No'])
feature_list['smoke'] = st.selectbox(label='Smoking', options=['Yes', 'No'])
feature_list['gluc'] = st.number_input('Enter Blood Glucose Level:', format='%.0f')
feature_list['cholesterol'] = st.number_input('Enter Cholesterol Level:', format='%.0f')
feature_list['systolic'] = st.number_input('Enter Systolic Blood Pressure:', format='%.0f')
feature_list['diastolic'] = st.number_input('Enter Diastolic Blood Pressure:', format='%.0f')

column = ['gluc', 'diastolic', 'systolic', 'alco', 'height', 'weight', 'active', 'gender', 'smoke', 'cholesterol', 'age']

def categorize_probability(prob):
    if prob <= 20:
        return 'Low Risk / Healthy'
    elif prob <= 55:
        return 'Moderate Risk'
    else:
        return 'High Risk'

def recommendation(rec):
    if rec == 'Low Risk / Healthy':
        return 'Schedule routine check-ups.'
    elif rec == 'Moderate Risk':
        return 'Seek Medical Advice for preventive measures.'
    else:
        return 'Urgently consult with a healthcare professional.'

if st.button('Predict'):
    # Calculate BMI
    bmi = feature_list['weight'] / ((feature_list['height'] / 100) ** 2)
    
    # Perform predictions
    df_feature_list = pd.DataFrame(data=np.array(list(feature_list.values())).reshape(1, -1), columns=feature_list.keys())
    df_feature_list = df_feature_list[column]

    final_features = preprocess(df_feature_list, scaler)
    print(final_features)
    
    prob = model.predict_proba(final_features)[:, 1] * 100
    text = categorize_probability(prob)
    text2 = recommendation(text)

    # Display BMI
    st.write(f'BMI: {bmi:.2f}')

    # Color and text based on risk level
    if text == 'Low Risk / Healthy':
        color = 'green'
    elif text == 'Moderate Risk':
        color = 'gold'
    else:
        color = 'red'

    st.markdown(f'<p style="background-color:{color}; color:black; padding:10px; border-radius:5px;">Risk Prediction: {prob[0]:.2f} % ({text}) {text2} </p>', unsafe_allow_html=True)
    
st.markdown("<div style='text-align: right; font-size: small;'>&copy; 2024 Data Geeks. All rights reserved.</div>", unsafe_allow_html=True)
