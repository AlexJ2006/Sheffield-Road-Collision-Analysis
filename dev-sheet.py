import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sbn


#Reading my CSV (Sheffield Specific Data)
sheffield_dataframe = pd.read_csv('Collision Data - Sheffield ONLY.csv')

print("")
print(sheffield_dataframe.isnull()) #Checking which of the columns are null


# Will now individually check each column to see which ones specifically are null

for column in sheffield_dataframe.columns.isnull:
    if sheffield_dataframe[column].isnull().any():
        print(f"{column}")
print("")


#I will now check for any outliers where applicable, using a scatter graph.

print(sheffield_dataframe.shape)

print(sheffield_dataframe['collision_adjusted_severity_slight'])

#Reading a new CSV so that I can remove the data from it.
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Updated.csv')

# Beginning imputation - local authority highway current COLUMN

mode_value = sheffield_dataframe_updated['local_authority_highway_current'].mode()[0] #Getting the modal (most common) value from the column

sheffield_dataframe_updated['local_authority_highway_current'] = (
    sheffield_dataframe_updated['local_authority_highway_current'] 
    .fillna(mode_value) #Filling the n/a spaces with the modal value.
)

sheffield_dataframe_updated['local_authority_highway_current'].isna().sum() #Returning the final total of the n/a values present within the column (0)


#Columns during data pre processing that contain null values

#location_easting_osgr
#location_northing_osgr
#longitude
#latitude
#local_authority_highway_current
#collision_adjusted_severity_serious
#collision_adjusted_severity_slight

#Need to also get the data types of the items

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Read data
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Cleaned.csv')

# Feature and label
X = sheffield_dataframe_updated[['weather_conditions']]
y = sheffield_dataframe_updated['collision_severity']

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=3
)

# ---------------------------------
# 1.12 Encode + standardise features
# ---------------------------------

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

x_train_encoded = encoder.fit_transform(x_train)
x_test_encoded = encoder.transform(x_test)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_encoded)
x_test_scaled = scaler.transform(x_test_encoded)

# ---------------------------------
# 1.13 Build SVM (linear kernel)
# ---------------------------------

svm_linear = SVC(
    C=1,
    kernel='linear',
    class_weight='balanced'
)

# ---------------------------------
# 1.14 Train
# ---------------------------------

svm_linear.fit(x_train_scaled, y_train)

# ---------------------------------
# 1.15 Test
# ---------------------------------

y_pred = svm_linear.predict(x_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))