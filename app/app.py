import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from predict import predict_survival

st.set_page_config(page_title="Titanic Survival Predictor", layout='centered')

st.title("Titanic Survival Prediction App")
st.write("Enter passenger information to estimate survival probability.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Number of siblings/spouces aboard", 0, 10, 0)
parch = st.number_input("Number of parents/childre aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

input_data = {
    'Pclass': pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
    "Name": "User, Mr. Example",
    "Ticket": "00000", 
    "Cabin": None,
    "PassengerId": 9999
}

if st.button('Predict Survival'):
    try:
        prediction, probability = predict_survival(input_data)

        if prediction == 1:
            st.success(f"Prediction: Survived")
        else:
            st.error(f"Prediction: Did Not Survive")

        st.write(f"Survival Probility: {probability:.2%}")

    except FileNotFoundError:
        st.error("No model found. Please! train the model first.")