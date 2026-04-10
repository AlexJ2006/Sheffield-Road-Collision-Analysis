import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Read the dataset
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Cleaned.csv')

feature_columns = ['weather_conditions']

X = sheffield_dataframe_updated[feature_columns]
y = sheffield_dataframe_updated['collision_severity']

# Train / test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=80
)

# Encoding categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

x_train_encoded = encoder.fit_transform(x_train)
x_test_encoded = encoder.transform(x_test)

# Standardising the features 
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_encoded)
x_test_scaled = scaler.transform(x_test_encoded)

#   linear SVC Model
svc_linear = SVC(
    C=1,
    kernel='linear',
    class_weight='balanced'
)

svc_linear.fit(x_train_scaled, y_train)

#   Asking for a prediction of Y
y_pred = svc_linear.predict(x_test_scaled)

#   Printing the accuracy of the model
print(accuracy_score(y_test, y_pred))

# import numpy as np
# import pandas as pd # Import pandas

# print("\n==============================")
# print(" DIABETES PREDICTION SYSTEM ")
# print("==============================\n")


# # ---------- SAFE INPUT FUNCTION ----------
# def get_value(prompt):
#     while True:
#         try:
#             return float(input(prompt))
#         except ValueError:
#             print("Invalid input. Please enter a numeric value.\n")


# # ---------- TAKE INPUT ----------
# Pregnancies = get_value("Enter number of Pregnancies: ")
# Glucose = get_value("Enter Glucose level: ")
# BloodPressure = get_value("Enter Blood Pressure value: ")
# SkinThickness = get_value("Enter Skin Thickness value: ")
# Insulin = get_value("Enter Insulin level: ")
# BMI = get_value("Enter BMI value: ")
# DiabetesPedigreeFunction = get_value("Enter Diabetes Pedigree Function value: ")
# Age = get_value("Enter Age: ")


# # ---------- CREATE ARRAY AND CONVERT TO DATAFRAME WITH FEATURE NAMES ----------
# user_data_array = np.array([[ # Create array first
#     Pregnancies,
#     Glucose,
#     BloodPressure,
#     SkinThickness,
#     Insulin,
#     BMI,
#     DiabetesPedigreeFunction,
#     Age
# ]])

# # Define column names based on the order of input features
# feature_names = [
#     'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
# ]

# # Convert to DataFrame with specified column names
# user_data = pd.DataFrame(user_data_array, columns=feature_names)


# # ---------- SCALE INPUT ----------
# user_data = scaler.transform(user_data)


# # ---------- PREDICT ----------
# prediction = svm_linear.predict(user_data)[0]


# # ---------- OUTPUT ----------
# print("\n==============================")
# print(" RESULT ")
# print("==============================")

# if prediction == 1:
#     print("Prediction: Diabetic")
# else:
#     print("Prediction: Not Diabetic")


# # ---------- PROBABILITY ----------
# if hasattr(svm_linear, "predict_proba"):
#     prob = svm_linear.predict_proba(user_data)[0][1]
#     print("Probability of Diabetes:", round(prob,3))

# print("\nDone.")

#Building an SVM Classifier (RBF Kernel)

SVM_classifier = SVC(C=1, kernel='poly',class_weight=None, gamma='scale')

SVM_classifier.fit(x_train_scaled, y_train)

#   Asking for a prediction of Y
y_pred = svc_linear.predict(x_test_scaled)

#   Printing the accuracy of the model
print(accuracy_score(y_test, y_pred))

