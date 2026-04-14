import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = fetch_openml('credit-g', version=1, as_frame=True)
df = data.frame

# Prepare data
y = df['class'].map({'good': 1, 'bad': 0})
df = df.drop('class', axis=1)
X = pd.get_dummies(df, drop_first=True)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# ---------------- UI ----------------
st.title("💳 Credit Scoring App")

st.write("Enter details to check creditworthiness")

# Inputs (simple demo inputs)
age = st.number_input("Age", 18, 100, 25)
credit_amount = st.number_input("Credit Amount", 100, 20000, 2000)
duration = st.number_input("Loan Duration (months)", 6, 72, 12)

# Predict button
if st.button("Predict"):

    # Create dummy input (same shape as training data)
    input_data = np.zeros((1, X.shape[1]))

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Good Credit")
    else:
        st.error("❌ Bad Credit")