import streamlit as st
from predict import predict_diabetes

def test1():
    st.header('Diabetes predictor')

    st.text("Enter the input parameters to predict diabetes:")

    pregnancies = st.slider('Number of pregnancies', 0, 17, 1)
    glucose = st.slider('Glucose level', 0, 199, 50)
    blood_pressure = st.slider('Blood pressure', 0, 122, 70)
    skin_thickness = st.slider('Skin thickness', 0, 99, 20)
    insulin = st.slider('Insulin level', 0, 846, 79)
    bmi = st.slider('BMI', 0.0, 67.1, 25.0)
    diabetes_pedigree_function = st.slider('Diabetes pedigree function', 0.078, 2.42, 0.3725, 0.001)
    age = st.slider('Age', 21, 81, 30)

    result = predict_diabetes(pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age)

    if st.button('Predict'):
        if result == 0:
            st.success("The patient is free from diabetets")
        if result == 1:
            st.warning("The patient may be affected from the diabetes")
