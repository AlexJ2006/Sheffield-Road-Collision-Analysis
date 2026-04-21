import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

st.title("SVM Collision Severity Analysis")

# Loading the data from the cleaned dataset file
df = pd.read_csv("../Sheffield Collision Data Cleaned.csv")

X = df[['weather_conditions']]
y = df['collision_severity']

# Train, Test, Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=80
)

# Encoding the Features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

x_train_encoded = encoder.fit_transform(x_train)
x_test_encoded = encoder.transform(x_test)

# Standardising the features
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_encoded)
x_test_scaled = scaler.transform(x_test_encoded)

# Building the SVM Classifier (Linear)
svm_linear = SVC(
    C=1,
    kernel='linear',
    class_weight='balanced'
)

# Training the SVM Classifier
svm_linear.fit(x_train_scaled, y_train)

# Testing the SVM Classifier
linear_pred = svm_linear.predict(x_test_scaled)
linear_accuracy = accuracy_score(y_test, linear_pred)

st.subheader("Linear Kernel Accuracy")
st.write(linear_accuracy)

# Building the SVM Classifier (RBF)
svm_rbf = SVC(
    C=1,
    kernel='rbf',
    gamma=0.001,
    class_weight='balanced'
)

# Training the RBF Classifier
svm_rbf.fit(x_train_scaled, y_train)

# Testing the RBF Classifier
rbf_pred = svm_rbf.predict(x_test_scaled)
rbf_accuracy = accuracy_score(y_test, rbf_pred)

st.subheader("RBF Kernel Accuracy")
st.write(rbf_accuracy)

# Building the SVM Classifier (Polynomial)
svm_poly = SVC(
    C=1,
    kernel='poly',
    gamma=0.001,
    degree=4,
    coef0=0.0,
    class_weight='balanced'
)

# Training the Polynomial Classifier
svm_poly.fit(x_train_scaled, y_train)

# Testing the Polynomial Classifier
poly_pred = svm_poly.predict(x_test_scaled)
poly_accuracy = accuracy_score(y_test, poly_pred)

st.subheader("Polynomial Kernel Accuracy")
st.write(poly_accuracy)

# Model Classification
st.subheader("Model Comparison")

comparison = pd.DataFrame({
    "Model": ["Linear", "RBF", "Polynomial"],
    "Accuracy": [linear_accuracy, rbf_accuracy, poly_accuracy]
})

st.dataframe(comparison)

# User Input Prediction
st.subheader("Predict Collision Severity")

weather = st.selectbox(
    "Weather Conditions",
    df['weather_conditions'].unique()
)

user_df = pd.DataFrame({
    "weather_conditions": [weather]
})

user_encoded = encoder.transform(user_df)
user_scaled = scaler.transform(user_encoded)

model_choice = st.selectbox(
    "Choose Model",
    ["Linear", "RBF", "Polynomial"]
)

if st.button("Predict"):

    if model_choice == "Linear":
        prediction = svm_linear.predict(user_scaled)

    elif model_choice == "RBF":
        prediction = svm_rbf.predict(user_scaled)

    else:
        prediction = svm_poly.predict(user_scaled)

    st.success(f"Predicted Collision Severity: {prediction[0]}")

# Plotting the Train/Test Data
st.subheader("Train vs Test Data Distribution")

fig, ax = plt.subplots()

ax.scatter(
    range(len(y_train)),
    y_train,
    label="Train Data"
)

ax.scatter(
    range(len(y_test)),
    y_test,
    label="Test Data"
)

ax.legend()

st.pyplot(fig)