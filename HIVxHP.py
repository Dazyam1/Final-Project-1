import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Health Predictor", layout="centered")

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

# Layout
st.title("🩺 Health Prediction Dashboard")
st.markdown("Predict the likelihood of **HIV** or **Hepatitis B** using health screening data.")

model_choice = st.sidebar.selectbox("Choose a Model", ["HIV Prediction", "Hepatitis B Prediction"])
show_metrics = st.sidebar.checkbox("Show HIV Model Metrics")

# --- HIV Prediction ---
if model_choice == "HIV Prediction":
    st.header("🧬 HIV Prediction Form")

    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Separated", "Widowed", "Other"])
    state = st.selectbox("State", nigerian_states)
    sexually_active = st.radio("Sexually Active?", ["Yes", "No"])
    screening_category = st.selectbox("Reason for HIV Screening", [
        "Routine ANC", "TB symptoms", "Partner positive", "HIV-exposed infant",
        "Other", "Positive parent", "Social contact", "General check-up",
        "High-risk sexual behavior"
    ])

    # Mappings
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

    if st.button("Predict HIV Likelihood"):
        prediction = hiv_model.predict(input_data)[0]
        if prediction == 1:
            st.markdown("<h3 style='color:red;'>There is a high likelihood you may be HIV positive.</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:blue;'>There is a low likelihood of HIV infection.</h3>", unsafe_allow_html=True)

    # Metrics (if checkbox selected)
    if show_metrics:
        st.subheader("📊 HIV Model Evaluation (Simulated)")

        try:
            df = pd.read_csv("your_cleaned_hiv_dataset.csv")  # Replace with actual path
            X = df.drop("HIV Test result", axis=1)
            y = df["HIV Test result"]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            y_pred = hiv_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.write(f"Accuracy Score: **{acc:.2f}**")

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading test data: {e}")

# --- Hepatitis B Prediction ---
elif model_choice == "Hepatitis B Prediction":
    st.header("🧪 Hepatitis B Prediction Form")

    sex = st.selectbox("Sex", ['Male', 'Female'])
    age = st.number_input("Age", min_value=0.1, max_value=120.0, step=0.1)
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Widowed', 'Separated', 'Divorced', 'Cohabiting'])
    state = st.selectbox("State", nigerian_states)
    hiv_result = st.selectbox("HIV Test Result", ['Positive', 'Negative'])
    sexually_active = st.selectbox("Sexually Active?", ['Yes', 'No'])
    tb_screening = st.selectbox("TB Screening", [
        'No signs', 'Cough, night sweat and fever', 'Cough ', 'Fever', 'Night sweat',
        'Cough', 'Cough, Fever, night sweat', 'Dry cough', 'Cough, Fever ',
        'Fever, Night sweat', 'Weight loss, Night sweat and cough',
        'Cough, weight loss', 'Cough, fever', 'Cough, night sweat'
    ])

    # Encode
    sex_map = {'Male': 0, 'Female': 1}
    marital_map = {v: i for i, v in enumerate(['Single', 'Married', 'Widowed', 'Separated', 'Divorced', 'Cohabiting'])}
    state_map = {v: i for i, v in enumerate(nigerian_states)}
    hiv_map = {'Positive': 1, 'Negative': 0}
    active_map = {'Yes': 1, 'No': 0}
    tb_map = {v: i for i, v in enumerate([
        'No signs', 'Cough, night sweat and fever', 'Cough ', 'Fever', 'Night sweat',
        'Cough', 'Cough, Fever, night sweat', 'Dry cough', 'Cough, Fever ',
        'Fever, Night sweat', 'Weight loss, Night sweat and cough',
        'Cough, weight loss', 'Cough, fever', 'Cough, night sweat'
    ])}

    input_data = np.array([[sex_map[sex], age, marital_map[marital_status],
                            state_map[state], hiv_map[hiv_result],
                            active_map[sexually_active], tb_map[tb_screening]]])

    if st.button("Predict Hepatitis B Likelihood"):
        prediction = hep_model.predict(input_data)[0]
        if prediction == 1:
            st.markdown("<h3 style='color:red;'>There is a high likelihood you may be positive for Hepatitis B.</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:blue;'>There is a low likelihood of Hepatitis B infection.</h3>", unsafe_allow_html=True)
