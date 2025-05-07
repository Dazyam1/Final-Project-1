# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# ------------------------
# LOAD MODELS
# ------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

hiv_model = load_model("hiv_prediction_model.pkl")
hepatitis_model = load_model("hepatitis_model_lr.pkl")

# ------------------------
# SIDEBAR
# ------------------------
st.sidebar.title("Prediction Model Selector")
model_choice = st.sidebar.radio("Choose a model", ["HIV Prediction", "Hepatitis B Prediction"])
show_metrics = st.sidebar.checkbox("Show HIV Model Metrics")

st.title("Health Risk Prediction App")

# ------------------------
# HIV FORM
# ------------------------
def hiv_form():
    st.subheader("HIV Prediction")
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", 0, 120, 25)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Separated", "Widowed", "Other"])
    state = st.selectbox("State", [f"State{i}" for i in range(1, 13)])
    sexually_active = st.radio("Sexually Active?", ["Yes", "No"])
    screening_category = st.selectbox("Screening Category", [
        "Routine ANC", "TB symptoms", "Partner positive", "HIV-exposed infant",
        "Other", "Positive parent", "Social contact", "General check-up",
        "High-risk sexual behavior"
    ])

    # Encoding
    sex_map = {"Male": 1, "Female": 0}
    marital_map = {v: i for i, v in enumerate(["Single", "Married", "Divorced", "Separated", "Widowed", "Other"])}
    state_map = {v: i for i, v in enumerate([f"State{i}" for i in range(1, 13)])}
    active_map = {"Yes": 1, "No": 0}
    screen_map = {
        "Routine ANC": 0, "TB symptoms": 1, "Partner positive": 2,
        "HIV-exposed infant": 3, "Other": 8, "Positive parent": 4,
        "Social contact": 5, "General check-up": 6, "High-risk sexual behavior": 7
    }

    input_data = np.array([[sex_map[sex], age, marital_map[marital_status],
                            state_map[state], active_map[sexually_active],
                            screen_map[screening_category]]])

    if st.button("Predict HIV"):
        prediction = hiv_model.predict(input_data)[0]
        st.success(f"Prediction: {'HIV Positive' if prediction == 1 else 'HIV Negative'}")

# ------------------------
# HEPATITIS FORM
# ------------------------
def hepatitis_form():
    st.subheader("Hepatitis B Prediction")
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

    # Input
    sex = st.selectbox('Sex', sex_options)
    age = st.number_input('Age', min_value=0.1, max_value=120.0, step=0.1)
    marital_status = st.selectbox('Marital Status', marital_options)
    state = st.selectbox('State', state_options)
    hiv_result = st.selectbox('HIV Test Result', hiv_result_options)
    sexually_active = st.selectbox('Sexually Active?', active_options)
    tb_screening = st.selectbox('TB Screening', tb_screening_options)

    # Mapping
    sex_map = {'Male': 0, 'Female': 1}
    marital_map = {v: i for i, v in enumerate(marital_options)}
    state_map = {v: i for i, v in enumerate(state_options)}
    hiv_map = {'Positive': 1, 'Negative': 0}
    active_map = {'Yes': 1, 'No': 0}
    tb_map = {v: i for i, v in enumerate(tb_screening_options)}

    input_data = np.array([[  
        sex_map[sex], age, marital_map[marital_status], state_map[state],
        hiv_map[hiv_result], active_map[sexually_active], tb_map[tb_screening]
    ]])

    if st.button("Predict Hepatitis B"):
        prediction = hepatitis_model.predict(input_data)
        result = "Positive for Hepatitis B" if prediction[0] == 1 else "Negative for Hepatitis B"
        st.success(f"Prediction: {result}")

# ------------------------
# METRICS FOR HIV MODEL
# ------------------------
def show_hiv_metrics():
    st.subheader("HIV Model Performance (on Test Data)")
    try:
        df = pd.read_csv("your_cleaned_hiv_dataset.csv")  # Replace with actual path
        X = df.drop("HIV Test result", axis=1)
        y = df["HIV Test result"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred = hiv_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"Accuracy Score: {acc:.2f}")

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading test data or displaying metrics: {e}")

# ------------------------
# ROUTING
# ------------------------
if model_choice == "HIV Prediction":
    hiv_form()
    if show_metrics:
        show_hiv_metrics()

elif model_choice == "Hepatitis B Prediction":
    hepatitis_form()
