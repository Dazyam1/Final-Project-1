import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('hepatitis_model_lr.pkl')

st.title("Hepatitis B Prediction App")

# Define input options based on your cleaned data
sex_options = ['Male', 'Female']
marital_options = ['Single', 'Married', 'Widowed', 'Separated', 'Divorced', 'Cohabiting']
state_options = ['Plateau', 'Lagos', 'Kaduna', 'Akwa Ibom', 'Bauchi', 'Ogun',
                 'Gombe', 'Kano', 'Benue', 'Enugu', 'Adamawa', 'Kogi']
hiv_result_options = ['Positive', 'Negative']
active_options = ['Yes', 'No']
tb_screening_options = [
    'No signs', 'Cough, night sweat and fever', 'Cough ', 'Fever', 'Night sweat',
    'Cough', 'Cough, Fever, night sweat', 'Dry cough', 'Cough, Fever ',
    'Fever, Night sweat', 'Weight loss, Night sweat and cough',
    'Cough, weight loss', 'Cough, fever', 'Cough, night sweat'
]

# Input fields
sex = st.selectbox('Sex', sex_options)
age = st.number_input('Age', min_value=0.1, max_value=120.0, step=0.1)
marital_status = st.selectbox('Marital Status', marital_options)
state = st.selectbox('State', state_options)
hiv_result = st.selectbox('HIV Test Result', hiv_result_options)
sexually_active = st.selectbox('Sexually Active?', active_options)
tb_screening = st.selectbox('TB Screening', tb_screening_options)

# Button
if st.button("Predict Hepatitis B"):
    # Convert inputs to the format your model expects (label-encoded or mapped)
    sex_map = {'Male': 0, 'Female': 1}
    marital_map = {v: i for i, v in enumerate(marital_options)}
    state_map = {v: i for i, v in enumerate(state_options)}
    hiv_map = {'Positive': 1, 'Negative': 0}
    active_map = {'Yes': 1, 'No': 0}
    tb_map = {v: i for i, v in enumerate(tb_screening_options)}

    # Build the input array
    input_data = np.array([[  
        sex_map[sex],
        age,
        marital_map[marital_status],
        state_map[state],
        hiv_map[hiv_result],
        active_map[sexually_active],
        tb_map[tb_screening]
    ]])

    # Prediction
    prediction = model.predict(input_data)

    # Show result
    result = "Positive for Hepatitis B" if prediction[0] == 1 else "Negative for Hepatitis B"
    st.success(f"Prediction: {result}")
