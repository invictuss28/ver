import streamlit as st
import pandas as pd
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
st.write("### Predict the likelihood of heart disease using patient details.")
st.write("Enter the patient's information below to get a prediction.")

# Layout for input fields
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=80, value=50, step=1)
        sex = st.radio("Sex", ["Male", "Female"])

    with col2:
        cholesterol = st.slider("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200, step=10)

    # Submit button
    submitted = st.form_submit_button("Submit")

if submitted:
    try:
        model, scaler, le = load_model()
        input_data = pd.DataFrame([[age, sex, cholesterol]], columns=["age", "sex", "cholesterol"])
        # Encode the sex column in the input data
        input_data['sex'] = le.transform(input_data['sex'].map({'Male': 'M', 'Female': 'F'}))
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
