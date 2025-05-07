import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set page config for a clean white look
st.set_page_config(page_title="Health Predictor", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: white;
    }
    .stSelectbox > div > div:first-child {
        font-weight: 500;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_model(path):
    return joblib.load(path)

hiv_model = load_model("hiv_prediction_model.pkl")
hep_model = load_model("hepatitis_model_lr.pkl")

# Common options
nigerian_states = [
    'Plateau', 'Lagos', 'Kaduna', 'Akwa Ibom', 'Bauchi', 'Ogun',
    'Gombe', 'Kano', 'Benue', 'Enugu', 'Adamawa', 'Kogi'
]

# Sidebar - vertical design
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/WHO_Logo.svg/1200px-WHO_Logo.svg.png", width=100)
    model_choice = st.radio("Choose Prediction Model", ["HIV Prediction", "Hepatitis B Prediction"])
    show_metrics = st.checkbox("Show HIV Model Metrics")

# HIV Prediction UI
if model_choice == "HIV Prediction":
    st.title("🧬 HIV Prediction")

    with st.form("hiv_form"):
        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox("Sex", ["Male", "Female"])
            age = st.slider("Age", 0, 120, 30)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Separated", "Widowed", "Other"])
            state = st.selectbox("State", nigerian_states)

        with col2:
            sexually_active = st.radio("Sexually Active?", ["Yes", "No"])
            screening_category = st.selectbox("Reason for HIV Screening", [
                "Routine ANC", "TB symptoms", "Partner positive", "HIV-exposed infant",
                "Other", "Positive parent", "Social contact", "General check-up",
                "High-risk sexual behavior"])

        submit_btn = st.form_submit_button("Predict HIV Likelihood")

        if submit_btn:
            # Mapping
            sex_map = {"Male": 1, "Female": 0}
            marital_map = {v: i for i, v in enumerate(["Single", "Married", "Divorced", "Separated", "Widowed", "Other"])}
            state_map = {v: i for i, v in enumerate(nigerian_states)}
            active_map = {"Yes": 1, "No": 0}
            screen_map = {
                "Routine ANC": 0, "TB symptoms": 1, "Partner positive": 2,
                "HIV-exposed infant": 3, "Other": 8, "Positive parent": 4,
                "Social contact": 5, "General check-up": 6, "High-risk sexual behavior": 7
            }

            input_data = np.array([[sex_map[sex], age, marital_map[marital_status],
                                    state_map[state], active_map[sexually_active],
                                    screen_map[screening_category]]])

            prediction = hiv_model.predict(input_data)[0]
            if prediction == 1:
                st.markdown("<h3 style='color:red;'>There is a high likelihood you may be HIV positive.</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color:blue;'>There is a low likelihood of HIV infection.</h3>", unsafe_allow_html=True)

    if show_metrics:
        st.subheader("Model Evaluation")
        try:
    # Load test dataset
    df_test = pd.read_csv("hiv_test_data.csv")  # File saved in your notebook step

    X_test = df_test.drop("HIV Test result", axis=1)
    y_test = df_test["HIV Test result"]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f"Accuracy Score: {acc:.2f}")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading or displaying metrics: {e}")

# Hepatitis B Prediction UI
elif model_choice == "Hepatitis B Prediction":
    st.title("🧪 Hepatitis B Prediction")

    with st.form("hep_form"):
        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox("Sex", ['Male', 'Female'])
            age = st.slider("Age", 1, 120, 30)
            marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Widowed', 'Separated', 'Divorced', 'Cohabiting'])
            state = st.selectbox("State", nigerian_states)

        with col2:
            hiv_result = st.selectbox("HIV Test Result", ['Positive', 'Negative'])
            sexually_active = st.selectbox("Sexually Active?", ['Yes', 'No'])
            tb_screening = st.selectbox("TB Screening", [
                'No signs', 'Cough, night sweat and fever', 'Cough ', 'Fever', 'Night sweat',
                'Cough', 'Cough, Fever, night sweat', 'Dry cough', 'Cough, Fever ',
                'Fever, Night sweat', 'Weight loss, Night sweat and cough',
                'Cough, weight loss', 'Cough, fever', 'Cough, night sweat'])

        submit_btn = st.form_submit_button("Predict Hepatitis Likelihood")

        if submit_btn:
            sex_map = {'Male': 0, 'Female': 1}
            marital_map = {v: i for i, v in enumerate(['Single', 'Married', 'Widowed', 'Separated', 'Divorced', 'Cohabiting'])}
            state_map = {v: i for i, v in enumerate(nigerian_states)}
            hiv_map = {'Positive': 1, 'Negative': 0}
            active_map = {'Yes': 1, 'No': 0}
            tb_map = {v: i for i, v in enumerate([
                'No signs', 'Cough, night sweat and fever', 'Cough ', 'Fever', 'Night sweat',
                'Cough', 'Cough, Fever, night sweat', 'Dry cough', 'Cough, Fever ',
                'Fever, Night sweat', 'Weight loss, Night sweat and cough',
                'Cough, weight loss', 'Cough, fever', 'Cough, night sweat'])}

            input_data = np.array([[sex_map[sex], age, marital_map[marital_status],
                                    state_map[state], hiv_map[hiv_result],
                                    active_map[sexually_active], tb_map[tb_screening]]])

            prediction = hep_model.predict(input_data)[0]
            if prediction == 1:
                st.markdown("<h3 style='color:red;'>There is a high likelihood you may be positive for Hepatitis B.</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color:blue;'>There is a low likelihood of Hepatitis B infection.</h3>", unsafe_allow_html=True)

