import streamlit as st
import pickle
import numpy as np


def load_model(path):
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data


data = load_model("model.pkl")

reg = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

countries = (
    "United States of America",
    "Other",
    "Germany",
    "United Kingdom of Great Britain and Northern Ireland",
    "India",
    "Ukraine",
    "France",
    "Canada",
    "Netherlands",
    "Brazil",
    "Spain",
    "Australia",
    "Italy",
    "Sweden",
    "Poland",
    "Switzerland",
    "Austria",
    "Russian Federation",
    "Norway",
    "Denmark",
    "Israel",
    "Portugal",
    "Czech Republic",
    "Belgium",
    "New Zealand",
)

educations = (
    "Bachelor's degree",
    "Master's degree",
    "No degree",
    "Post grad",
)


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("We need some info to predict the salary")

    country = st.selectbox("Country", countries)
    education = st.selectbox("Edication Level", educations)

    experience = st.slider("Years of Professional Experience", 0, 50, 3, 1)

    if st.button("Predict Salary"):
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary = reg.predict(X)
        st.subheader(f"Predicted Salary is ${salary[0]:.2f}")
