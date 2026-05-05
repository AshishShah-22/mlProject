import streamlit as st
import pandas as pd
import pickle


@st.cache_resource
def load_objects():
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("artifacts/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor

model, preprocessor = load_objects()

st.title(" Model Test (Streamlit)")


gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
education = st.selectbox("Parental Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing = st.number_input("Writing Score", min_value=0, max_value=100, value=50)


if st.button("Predict"):
    try:
        data = pd.DataFrame({
            "gender": [gender],
            "race/ethnicity": [race],
            "parental level of education": [education],
            "lunch": [lunch],
            "test preparation course": [test_course],
            "reading score": [reading],
            "writing score": [writing]
        })

        st.write(" Input Data:", data)

        data_scaled = preprocessor.transform(data)
        pred = model.predict(data_scaled)

        st.success(f" Prediction: {pred[0]}")

    except Exception as e:
        st.error(f" Error: {e}")