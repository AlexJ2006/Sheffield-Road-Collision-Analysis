import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np

st.title("SVM Collision Severity Analysis")

# Loading the clean dataset
df = pd.read_csv("../Sheffield Collision Data Cleaned.csv")

# Loading in multiple features
x = df[['weather_conditions', 'speed_limit', 'number_of_vehicles']]
y = df['collision_severity']

# Splitting the dataset
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, train_size=0.70, random_state=80
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.50, random_state=80
)

# Separate categorical and numerical features
categorical_features = ['weather_conditions']
numerical_features = ['speed_limit', 'number_of_vehicles']

# Encoding the categorical features
# This isn't needed for the numerical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

x_train_cat = encoder.fit_transform(x_train[categorical_features])
x_val_cat = encoder.transform(x_val[categorical_features])
x_test_cat = encoder.transform(x_test[categorical_features])

# Get numerical features
x_train_num = x_train[numerical_features].values
x_val_num = x_val[numerical_features].values
x_test_num = x_test[numerical_features].values

# Combine categorical and numerical features

x_train_combined = np.hstack((x_train_cat, x_train_num))
x_val_combined = np.hstack((x_val_cat, x_val_num))
x_test_combined = np.hstack((x_test_cat, x_test_num))

# Standardise features
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_combined)
x_val_scaled = scaler.transform(x_val_combined)
x_test_scaled = scaler.transform(x_test_combined)

# Check shapes
st.write("Train shape:", x_train_scaled.shape, y_train.shape)
st.write("Validation shape:", x_val_scaled.shape, y_val.shape)
st.write("Test shape:", x_test_scaled.shape, y_test.shape)

