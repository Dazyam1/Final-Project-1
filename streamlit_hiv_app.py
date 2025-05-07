import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import joblib

# Load the model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Load HIV model
model = load_model("hiv_prediction_model.pkl")

# HIV Input Form
def hiv_form():
    st.subheader("HIV Prediction Input")

    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", 0, 120, 25)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Separated", "Widowed", "Other"])
    state = st.selectbox("State", ["State1", "State2", "State3", "State4", "State5", "State6", "State7", "State8", "State9", "State10", "State11", "State12"])
    sexually_active = st.radio("Sexually Active?", ["Yes", "No"])
    screening_category = st.selectbox("Screening Category", [
        "Routine ANC", "TB symptoms", "Partner positive", "HIV-exposed infant",
        "Other", "Positive parent", "Social contact", "General check-up",
        "High-risk sexual behavior"
    ])

    # Encode inputs
    sex_map = {"Male": 1, "Female": 0}
    marital_map = {v: i for i, v in enumerate(["Single", "Married", "Divorced", "Separated", "Widowed", "Other"])}
    state_map = {v: i for i, v in enumerate([
        "State1", "State2", "State3", "State4", "State5", "State6",
        "State7", "State8", "State9", "State10", "State11", "State12"
    ])}
    active_map = {"Yes": 1, "No": 0}
    screen_map = {
        "Routine ANC": 0, "TB symptoms": 1, "Partner positive": 2,
        "HIV-exposed infant": 3, "Other": 8, "Positive parent": 4,
        "Social contact": 5, "General check-up": 6, "High-risk sexual behavior": 7
    }

    input_features = np.array([[sex_map[sex], age, marital_map[marital_status],
                                state_map[state], active_map[sexually_active],
                                screen_map[screening_category]]])

    return input_features

# Sidebar control
st.title("Health Prediction Dashboard")
model_choice = st.sidebar.radio("Select Prediction Model", ["HIV"])
show_metrics = st.sidebar.checkbox("Show Model Metrics")

# Run prediction
if model_choice == "HIV":
    input_data = hiv_form()
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"Prediction: {'HIV Positive' if prediction == 1 else 'HIV Negative'}")

    # Optionally show model evaluation metrics
    if show_metrics:
        st.subheader("Model Performance (on Test Data)")

        try:
            # You must load your actual test data here:
            # X_test, y_test = load from file
            # For demo purposes, we simulate dummy data:
            from sklearn.model_selection import train_test_split
            import pandas as pd

            # Load a copy of your training dataset to simulate test scoring
            df = pd.read_csv("your_cleaned_hiv_dataset.csv")  # Replace with your actual dataset path
            X = df.drop("HIV Test result", axis=1)
            y = df["HIV Test result"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.write(f"Accuracy Score: {acc:.2f}")

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading test data or displaying metrics: {e}")
