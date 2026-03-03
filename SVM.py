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