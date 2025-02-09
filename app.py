import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Cargar el modelo y los preprocesadores
with open('models/svm_model_adult.pck', 'rb') as f:
    scaler, model, label_encoders, feature_names = pickle.load(f)

# Título de la app
st.title("Predicción de Ingresos con Machine Learning")
st.write("Ingrese la información para predecir si la persona gana más o menos de 50K al año.")

# Crear formulario de entrada de datos
age = st.number_input("Edad", min_value=18, max_value=90, value=30)
workclass = st.selectbox("Workclass", label_encoders['WORKCLASS'].classes_)
education = st.selectbox("Educación", label_encoders['EDUCATION'].classes_)
education_num = st.slider("Años de educación", min_value=1, max_value=16, value=10)
marital_status = st.selectbox("Estado Civil", label_encoders['MARITAL-STATUS'].classes_)
occupation = st.selectbox("Ocupación", label_encoders['OCCUPATION'].classes_)
relationship = st.selectbox("Relación familiar", label_encoders['RELATIONSHIP'].classes_)
sex = st.selectbox("Género", label_encoders['SEX'].classes_)
hours_per_week = st.number_input("Horas trabajadas por semana", min_value=1, max_value=100, value=40)
capital_gain = st.number_input("Ganancias de capital", min_value=0, value=0)
capital_loss = st.number_input("Pérdidas de capital", min_value=0, value=0)

# Convertir las entradas categóricas a numéricas
workclass_encoded = label_encoders['WORKCLASS'].transform([workclass])[0]
education_encoded = label_encoders['EDUCATION'].transform([education])[0]
marital_status_encoded = label_encoders['MARITAL-STATUS'].transform([marital_status])[0]
occupation_encoded = label_encoders['OCCUPATION'].transform([occupation])[0]
relationship_encoded = label_encoders['RELATIONSHIP'].transform([relationship])[0]
sex_encoded = label_encoders['SEX'].transform([sex])[0]

# Crear un array con los valores ingresados
input_data = np.array([[age, workclass_encoded, education_encoded, education_num, 
                        marital_status_encoded, occupation_encoded, relationship_encoded, 
                        sex_encoded, capital_gain, capital_loss, hours_per_week]])

# Escalar los datos de entrada
input_data_scaled = scaler.transform(input_data)

# Predicción
if st.button("Predecir Ingreso"):
    prediction = model.predict(input_data_scaled)
    resultado = "Más de 50K" if prediction[0] == 1 else "Menos de 50K"
    st.success(f"Predicción: {resultado}")
