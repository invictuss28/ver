import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset (REPLACE THIS WITH YOUR ACTUAL DATA LOADING)
data = {'age': [30, 40, 50, 60, 70, 35, 45, 55, 65, 75],
        'sex': ['M', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'M'], 
        'cholesterol': [180, 220, 190, 250, 210, 195, 215, 185, 240, 205],
        'target': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

# Encode the sex column using LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale your data using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Load and train model
@st.cache_data
def load_model():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, le

# Streamlit app
st.title("Heart Disease Prediction")
st.write("### Predict the likelihood of heart disease using patient information.")
st.write("Please input the details in the sidebar and view the prediction below.")

# Sidebar inputs
st.sidebar.header("Patient Details")
age = st.sidebar.slider("Age", 20, 80, 50)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
cholesterol = st.sidebar.number_input("Cholesterol Level (mg/dL)", 100, 400, 200)

# Prediction
try:
    model, scaler, le = load_model()
    input_data = pd.DataFrame([[age, sex, cholesterol]], columns=["age", "sex", "cholesterol"])
    # Encode the sex column in the input data
    input_data['sex'] = le.transform(input_data['sex'])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    # Display prediction
    st.write("### Prediction Results")
    if prediction[0] == 1:
        st.error("The model predicts that the patient is **at risk** of heart disease.")
    else:
        st.success("The model predicts that the patient is **not at risk** of heart disease.")
except Exception as e:
    st.error(f"An error occurred: {e}")
