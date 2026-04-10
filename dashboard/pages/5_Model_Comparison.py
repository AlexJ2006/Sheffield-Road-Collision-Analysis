import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression

# PAGE CONFIG

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("📊 Model Comparison")

# LOAD DATA (MATCHING YOUR PROJECT)

@st.cache_data
def load_data():
    return pd.read_csv("../Sheffield Collision Data Cleaned.csv")

df = load_data()

# FEATURE ENGINEERING (MATCH YOUR CODE)

df['hour'] = pd.to_datetime(df['time'], errors='coerce').dt.hour

feature_columns = ['weather_conditions', 'road_type', 'light_conditions','speed_limit',
                   'number_of_vehicles', 'road_surface_conditions', 'junction_detail',
                   'junction_control', 'urban_or_rural_area', 'day_of_week', 'hour']

X = df[feature_columns]
y = df['collision_severity']

categorical_features = ['weather_conditions', 'road_type', 'light_conditions',
                        'road_surface_conditions', 'junction_detail', 'junction_control',
                        'urban_or_rural_area', 'day_of_week']

numerical_features = ['speed_limit', 'number_of_vehicles', 'hour']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

X_train_fe = preprocessor.fit_transform(X_train)
X_test_fe = preprocessor.transform(X_test)

# MODELS

# Default models
knn_default = KNeighborsClassifier(n_neighbors=4)
svm_default = SVC()
reg = LinearRegression()

knn_default.fit(X_train_fe, y_train)
svm_default.fit(X_train_fe, y_train)
reg.fit(X_train_fe, y_train)
y_pred_reg = reg.predict(X_test_fe)

# TUNING

# KNN tuning
knn_params = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}

knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_grid.fit(X_train_fe, y_train)
knn_tuned = knn_grid.best_estimator_

# SVM tuning
svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}

svm_grid = GridSearchCV(SVC(), svm_params, cv=5)
svm_grid.fit(X_train_fe, y_train)
svm_tuned = svm_grid.best_estimator_

# METRICS

def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def regression_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred)
    }

# Collect results
results = []

def add_classification_result(name, y_true, y_pred):
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    })

# KNN
add_classification_result("KNN (Default)", y_test, knn_default.predict(X_test_fe))
add_classification_result("KNN (Tuned)", y_test, knn_tuned.predict(X_test_fe))

# SVM
add_classification_result("SVM (Default)", y_test, svm_default.predict(X_test_fe))
add_classification_result("SVM (Tuned)", y_test, svm_tuned.predict(X_test_fe))

# Regression
metrics_reg = regression_metrics(y_test, y_pred_reg)
metrics_reg["Model"] = "Regression"
results.append(metrics_reg)

results_df = pd.DataFrame(results)

# DISPLAY TABLE

st.subheader("Model Performance Table")
st.dataframe(results_df.set_index("Model"))

# VISUAL COMPARISON

st.subheader("Performance Comparison")

metric_choice = st.selectbox(
    "Select metric to compare",
    ["Accuracy", "Precision", "Recall", "F1 Score", "RMSE", "R²"]
)

if metric_choice in results_df.columns:
    chart_df = results_df[["Model", metric_choice]].dropna()

    st.bar_chart(chart_df.set_index("Model"))

# BEST MODEL

st.subheader("🏆 Best Model")

if "Accuracy" in results_df.columns:
    best_model = results_df.sort_values("Accuracy", ascending=False).iloc[0]
    st.success(f"Best classification model: {best_model['Model']}")

if "R²" in results_df.columns:
    best_reg = results_df.sort_values("R²", ascending=False).iloc[0]
    st.info(f"Best regression model: {best_reg['Model']}")

# Logistics Regresion   

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_fe, y_train)

add_classification_result(
    "Logistic Regression",
    y_test,
    log_reg.predict(X_test_fe)
)

