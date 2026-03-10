import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

st.title("SVM Collision Severity Analysis")

# Loading the data from the cleaned dataset file

df = pd.read_csv("../Sheffield Collision Data Cleaned.csv")

x = df[['weather_conditions']]
y = df['collision_severity']

# Train (70%) and Temporary (30%)

x_train, x_temp, y_train, y_temp = train_test_split(
    x, y,
    train_size=0.70,
    random_state=80
)

# Validation (15%) and Test (15%)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp,
    test_size=0.50,
    random_state=80
)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

x_train_encoded = encoder.fit_transform(x_train)
x_val_encoded = encoder.transform(x_val)
x_test_encoded = encoder.transform(x_test)

# Checking the shape of the sets
print(x_train.shape)
print(y_train.shape)

# Check shapes of validation sets
print(x_val.shape)
print(y_val.shape)

# Check shapes of test sets
print(x_test.shape)
print(y_test.shape)