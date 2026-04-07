
# 1. IMPORTS
# =========================
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from termcolor import colored
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FormatStrFormatter
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve

# 2. DATA LOADING
# =========================

#Reading my CSV (Sheffield Specific Data)
sheffield_dataframe = pd.read_csv('Collision Data - Sheffield ONLY.csv')

# 3. PREPROCESSING
# =========================

# SANITY CHECKING
print("")
sheffield_dataframe.isnull().sum()
print("")
print('==================================================================')
print(colored("Below are the columns that contain null data:", 'red'))
print("")
for column in sheffield_dataframe.columns:
    if sheffield_dataframe[column].isnull().any():
        print(f"{column}")
print("")
print('==================================================================')
print("")

#-----------------------------------------              Local_authority_highway_current               ---------------------------------------------------

sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Updated.csv')

local_authority_highway_current_dataType = sheffield_dataframe_updated['local_authority_highway_current'].dtype
print(f'Local Authority Highway Current (DataType):', local_authority_highway_current_dataType)
print("")
# INITIAL Histogram
#Drawing a histogram to view the which data is present in the column.

fig, ax = plt.subplots(figsize=(8, 8))

sbn.histplot(sheffield_dataframe_updated['local_authority_highway_current'])
ax.set_title('Values Present - Initial')
plt.show()

# The graph showed that the only highway data available for the entirety of the Sheffield dataset.
# Was E08000019
# Therefore, I can impute that the results in this column that are N/A can be filled with this value.

#Counting the initial number of N/A rows in the column.
local_authority_highway_current_na_sum = sheffield_dataframe_updated['local_authority_highway_current'].isna().sum()
print(f'Initial total N/A Values:', local_authority_highway_current_na_sum)

#This number is 5314.
#After looking over the dataset briefly, this seems accurate.

local_authority_highway_current_fill = 'E08000019' #The value that needs to fill the spaces
sheffield_dataframe_updated['local_authority_highway_current'] = (sheffield_dataframe_updated['local_authority_highway_current'].fillna(local_authority_highway_current_fill)) #Filling the N/A spaces with this value

# UPDATED HISTOGRAM
fig, ax = plt.subplots(figsize=(8, 8))

sbn.histplot(sheffield_dataframe_updated['local_authority_highway_current'])
ax.set_title('Values Present - Current')
plt.show()

#Re-printing the value to ensure the changes have been made
local_authority_highway_current_na_sum_2 = sheffield_dataframe_updated['local_authority_highway_current'].isna().sum()
print(f'Current total N/A Values:', local_authority_highway_current_na_sum_2)
print("")

#-------------------------------------------                longitude + latitude                --------------------------------------------------------------------

#HISTOGRAM - INITIAL LATITUDE
fig, ax = plt.subplots(figsize=(8,6))
sbn.histplot(sheffield_dataframe_updated['latitude'], bins=50)
ax.set_title("Latitude - INTIIAL")
plt.show()

#HISTOGRAM - INTIIAL LONGITUDE
fig, ax = plt.subplots(figsize=(8,6))
sbn.histplot(sheffield_dataframe_updated['longitude'], bins=50)
ax.set_title("Longitude - INITIAL")
plt.show()

#The graph shows that there are some outliers.

# Calculating mean values
latitude_mean = sheffield_dataframe_updated['latitude'].mean()      #For Latitude
longitude_mean = sheffield_dataframe_updated['longitude'].mean()        #And Longitude

# Filling the N/A Values with the mean
sheffield_dataframe_updated['latitude'] = sheffield_dataframe_updated['latitude'].fillna(latitude_mean)     #Latitude
sheffield_dataframe_updated['longitude'] = sheffield_dataframe_updated['longitude'].fillna(longitude_mean)      #Longitude

# Printing updated N/A counts
#For latitude
latitude_na_sum = sheffield_dataframe_updated['latitude'].isna().sum()  
print(f"Updated N/A Values - Latitude: {latitude_na_sum}")
#And longitude
longitude_na_sum = sheffield_dataframe_updated['longitude'].isna().sum()
print(f"Updated N/A Values - Longitude: {longitude_na_sum}")

#HISTOGRAM - UPDATED LATITUDE
fig, ax = plt.subplots(figsize=(8,6))
sbn.histplot(sheffield_dataframe_updated['latitude'], bins=50)
ax.set_title("Latitude - UPDATED")
plt.show()

#HISTOGRAM - UPDATED LONGITUDE
fig, ax = plt.subplots(figsize=(8,6))
sbn.histplot(sheffield_dataframe_updated['longitude'], bins=50)
ax.set_title("Longitude - UPDATED")
plt.show()

#-------------------------------------------                Location Data               --------------------------------------------------------------------

print('==================================================================')
print("")
#Easting Data Type
location_easting_osgr_datatype = sheffield_dataframe_updated['location_easting_osgr'].dtype
print(f'Location Easting (DataType):', location_easting_osgr_datatype)      #Checking the data type of easting
#Northing Data Type
location_northing_osgr_datatype = sheffield_dataframe_updated['location_northing_osgr'].dtype
print(f'Location Northing (DataType):', location_northing_osgr_datatype)        #Checking the data type of northing
print("")
#Location - Northing NA Value Count
location_northing_na_sum = sheffield_dataframe_updated['location_northing_osgr'].isna().sum()
print(f'Initial total N/A Values - NORTHING:', location_northing_na_sum)        #Checking the N/A values of northing
#Total rows for northing
total_rows_northing = len(sheffield_dataframe_updated['location_northing_osgr'])
print(f"TOTAL ROWS - NORTHING: ", total_rows_northing)      #Checking the total length of the northing column
#Location - Easting NA Value Count
print("")
location_easting_na_sum = sheffield_dataframe_updated['location_easting_osgr'].isna().sum()
print(f'Initial total N/A Values - EASTING:', location_easting_na_sum)      #Checking the N/A values of easting
#Total rows for Easting
total_rows_easting = len(sheffield_dataframe_updated['location_easting_osgr'])
print(f"TOTAL ROWS - EASTING: ", total_rows_easting)        #Checking the total rows for easting
print("")

#HISTOGRAM - TOTAL PRESENT VALUES - NORTHING

fig, ax = plt.subplots(figsize=(8, 8))
plt.xlim(385000, 405000)

sbn.histplot(sheffield_dataframe_updated['location_northing_osgr'])
ax.set_title('Values Present - Initial - Northing')
plt.show()

#HISTOGRAM - TOTAL PRESENT VALUES - EASTING

fig, ax = plt.subplots(figsize=(8, 8))
plt.xlim(430000, 450000)

sbn.histplot(sheffield_dataframe_updated['location_easting_osgr'])
ax.set_title('Values Present - Initial - Easting')
plt.show()


#Filling the N/A values
easting_mean = sheffield_dataframe_updated['location_easting_osgr'].mean()      #Generating Easting Mean Value
northing_mean = sheffield_dataframe_updated['location_northing_osgr'].mean()        #Generating Northing Mean Value

#Updating the DataSet
#For northing
sheffield_dataframe_updated['location_northing_osgr'] = (
    sheffield_dataframe_updated['location_northing_osgr'].fillna(northing_mean)
)
#For easting
sheffield_dataframe_updated['location_easting_osgr'] = (
    sheffield_dataframe_updated['location_easting_osgr'].fillna(easting_mean)
)

#Printing the updated N/A values for NORTHING
location_northing_na_sum_2 = sheffield_dataframe_updated['location_northing_osgr'].isna().sum()
print(f'Current total N/A Values - NORTHING:', location_northing_na_sum_2)
#Same for Easting
location_easting_na_sum_2 = sheffield_dataframe_updated['location_easting_osgr'].isna().sum()
print(f'Current total N/A Values - EASTING:', location_easting_na_sum_2)
print("")


#HISTOGRAM - UPDATED - NORTHING
fig, ax = plt.subplots(figsize=(8, 8))
plt.xlim(385000, 405000)

sbn.histplot(sheffield_dataframe_updated['location_northing_osgr'])
ax.set_title('Values Present - Current - Northing')
plt.show()

#HISTOGRAM - UPDATED - EASTING
fig, ax = plt.subplots(figsize=(8, 8))
plt.xlim(430000, 450000)

sbn.histplot(sheffield_dataframe_updated['location_easting_osgr'])
ax.set_title('Values Present - Current - Easting')
plt.show()

print('==================================================================')

#-------------------------------------------               SERIOUS COLLISIONS               --------------------------------------------------------------------

#Collision Serious
collision_adjusted_severity_serious_datatype = sheffield_dataframe_updated['collision_adjusted_severity_serious'].dtype
print(f'Collision Serious (DataType):', collision_adjusted_severity_serious_datatype)      #Checking the data type of the serious collisions

#Location - Easting NA Value Count
print("")
collision_serious_na_sum = sheffield_dataframe_updated['collision_adjusted_severity_serious'].isna().sum()
print(f'Initial N/A Values - Serious Collisions', collision_serious_na_sum)      #Checking the N/A values of easting
print("")

#HISTOGRAM - INITIAL VALUES
fig, ax = plt.subplots(figsize=(8, 8))

sbn.histplot(sheffield_dataframe_updated['collision_adjusted_severity_serious'])
ax.set_title('INTIIAL VALUES - SERIOUS COLLISIONS: ')
plt.show()

serious_collisions = sheffield_dataframe_updated['collision_adjusted_severity_serious'].mode()[0]

sheffield_dataframe_updated['collision_adjusted_severity_serious'] = (
    sheffield_dataframe_updated['collision_adjusted_severity_serious']
    .fillna(serious_collisions)
)

#HISTOGRAM - INITIAL VALUES
fig, ax = plt.subplots(figsize=(8, 8))

sbn.histplot(sheffield_dataframe_updated['collision_adjusted_severity_serious'])
ax.set_title('CURRENT VALUES - SERIOUS COLLISIONS: ')
plt.show()

#UPDATED N/A COUNT
collision_serious_na_sum_2 = sheffield_dataframe_updated['collision_adjusted_severity_serious'].isna().sum()
print(f'CURRENT N/A Values - Serious Collisions', collision_serious_na_sum_2)
print("")

#-------------------------------------------               SLIGHT COLLISIONS               --------------------------------------------------------------------
print('==================================================================')
#Collision Slight
collision_adjusted_severity_slight_datatype = sheffield_dataframe_updated['collision_adjusted_severity_slight'].dtype
print(f'Collision slight (DataType):', collision_adjusted_severity_slight_datatype)      #Checking the data type of the slight collisions

#Location - Easting NA Value Count
print("")
collision_slight_na_sum = sheffield_dataframe_updated['collision_adjusted_severity_slight'].isna().sum()
print(f'Initial N/A Values - Slight Collisions', collision_slight_na_sum)      #Checking the N/A values of easting
print("")

#HISTOGRAM - INITIAL VALUES
fig, ax = plt.subplots(figsize=(8, 8))

sbn.histplot(sheffield_dataframe_updated['collision_adjusted_severity_slight'])
ax.set_title('INTIIAL VALUES - SLIGHT COLLISIONS: ')
plt.show()

slight_collisions = sheffield_dataframe_updated['collision_adjusted_severity_slight'].mode()[0]

sheffield_dataframe_updated['collision_adjusted_severity_slight'] = (
    sheffield_dataframe_updated['collision_adjusted_severity_slight']
    .fillna(slight_collisions)
)

#HISTOGRAM - INITIAL VALUES
fig, ax = plt.subplots(figsize=(8, 8))

sbn.histplot(sheffield_dataframe_updated['collision_adjusted_severity_slight'])
ax.set_title('CURRENT VALUES - SLIGHT COLLISIONS: ')
plt.show()

#UPDATED N/A COUNT
collision_slight_na_sum_2 = sheffield_dataframe_updated['collision_adjusted_severity_slight'].isna().sum()
print(f'CURRENT N/A Values - Slight Collisions', collision_slight_na_sum_2)

#Setting the severity_slight column to return 0 or 1.
sheffield_dataframe_updated["collision_adjusted_severity_serious"] = (
    sheffield_dataframe_updated["collision_adjusted_severity_serious"]
        .astype(int)
        .map({0: "Not serious", 1: "Serious"})
)
#Same for the slight section
sheffield_dataframe_updated["collision_adjusted_severity_slight"] = (
    sheffield_dataframe_updated["collision_adjusted_severity_slight"]
        .astype(int)
        .map({0: "Not slight", 1: "Slight"})
)

#-----------------------------------------              FINAL SANITY CHECK               ---------------------------------------------------

# SANITY CHECKING
print("")
sheffield_dataframe_updated.isnull().sum()
print("")
print('==================================================================')
print(colored("Below are the columns that contain null data:", 'red'))
print("")
for column in sheffield_dataframe_updated.columns:
    if sheffield_dataframe_updated[column].isnull().any():
        print(f"{column}")
print("")
print('==================================================================')
print("")

#-----------------------------------------              ADDITIONAL GRAPHS FOR HIGHER MARKS              ---------------------------------------------------

#local_authority_highway_current                BAR CHART
sheffield_dataframe_updated['local_authority_highway_current'].value_counts().plot(kind="bar")
plt.title("Number Of Collisions In The Local Area")
plt.show()

#local_authority_highway_current                COUNT PLOT
sbn.countplot(
    data=sheffield_dataframe_updated.dropna(
        subset=["local_authority_highway_current"]
    ),
    x="local_authority_highway_current",
)
plt.title("Number Of Collisions - Local Area")
plt.show()

#collision_adjusted_severity_serious            COUNT PLOT
sbn.countplot(
    data=sheffield_dataframe_updated.dropna(
        subset=["collision_adjusted_severity_serious"]
    ),
    x="collision_adjusted_severity_serious",
)
plt.title("Count Plot - Serious Collisions")
plt.show()

#latitude                                       BOX PLOT
sbn.boxplot(
    data=sheffield_dataframe_updated,
    y='latitude',               
    color='skyblue',       
    linewidth=1,                 
    fliersize=2,              
    flierprops=dict(marker = "o", markerfacecolor= (0.7, 0.2, 0.4), markersize= 5, alpha= 0.7),
    medianprops=dict(color='black', linewidth= 2)
)
plt.title("Box Plot - Hilighting Latitudinal Outliers")
plt.show()

#latitude                                       KDE PLOT
sbn.kdeplot(data=sheffield_dataframe_updated["latitude"])
plt.title("KDE estimating latitudinal values")
plt.show()

#longitude                                      BOX PLOT
sbn.boxplot(
    data=sheffield_dataframe_updated,
    y='longitude',               
    color='skyblue',       
    linewidth=1,                 
    fliersize=2,              
    flierprops=dict(marker = "o", markerfacecolor= (0.7, 0.2, 0.4), markersize= 5, alpha= 0.7),
    medianprops=dict(color='black', linewidth= 2)
)
plt.title("Box Plot - Hilighting Longitudinal Outliers")
plt.show()

#longitude                                       KDE PLOT
sbn.kdeplot(data=sheffield_dataframe_updated["longitude"])
plt.title("KDE estimating longitudinal values")
plt.show()

#location_easting_osgr                           BOX PLOT
sbn.boxplot(
    data=sheffield_dataframe_updated,
    y='location_easting_osgr',               
    color='skyblue',       
    linewidth=1,                 
    fliersize=2,              
    flierprops=dict(marker = "o", markerfacecolor= (0.7, 0.2, 0.4), markersize= 5, alpha= 0.7),
    medianprops=dict(color='black', linewidth= 2)
)
plt.title("Box Plot - Easting Values")
plt.show()

#location_northing_osgr                          BOX PLOT
sbn.boxplot(
    data=sheffield_dataframe_updated,
    y='location_northing_osgr',               
    color='skyblue',       
    linewidth=1,                 
    fliersize=2,              
    flierprops=dict(marker = "o", markerfacecolor= (0.7, 0.2, 0.4), markersize= 5, alpha= 0.7),
    medianprops=dict(color='black', linewidth= 2)
)
plt.title("Box Plot - Northing Values")
plt.show()

#longitude AGAINST latitude                      SCATTER PLOT
sbn.scatterplot(
    data=sheffield_dataframe_updated,
    x='longitude',
    y='latitude'
)
plt.title("Distribution of collisions (geographical) - Latitude VS Longitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

#easting AGAINST northing                       SCATTER PLOT
sbn.scatterplot(
    data=sheffield_dataframe_updated,
    x='location_northing_osgr',
    y='location_easting_osgr'
)
plt.title("Collision Locations")
plt.xlabel("Northing")
plt.ylabel("Easting")
plt.show()

sheffield_dataframe_updated.to_csv(
    "Sheffield Collision Data Cleaned.csv",
    index=False
)

# =========================
# 4. FEATURE ENGINEERING


df = pd.read_csv('Sheffield Collision Data Cleaned.csv')

# -------------------------------------------------------
# TASK A: MULTICLASS — predict collision_severity
# (Slight / Serious / Fatal)
# -------------------------------------------------------

multiclass_features = [
    'weather_conditions', 'road_surface_conditions', 'light_conditions',
    'speed_limit', 'number_of_vehicles', 'number_of_casualties',
    'urban_or_rural_area', 'day_of_week', 'junction_detail', 'road_type'
]

mc_df = df[multiclass_features + ['collision_severity']].dropna()

le_mc = LabelEncoder()
for col in mc_df.select_dtypes(include='object').columns:
    mc_df[col] = le_mc.fit_transform(mc_df[col].astype(str))

X_mc = mc_df[multiclass_features]
y_mc = mc_df['collision_severity']

# Train / val / test split (60/20/20)

X_temp, X_test_mc, y_temp, y_test_mc = train_test_split(
    X_mc, y_mc, test_size=0.2, random_state=42, stratify=y_mc)
X_train_mc, X_val_mc, y_train_mc, y_val_mc = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f'Multiclass split — train: {len(X_train_mc)}, val: {len(X_val_mc)}, test: {len(X_test_mc)}')

scaler_mc = StandardScaler()
X_train_mc_s = scaler_mc.fit_transform(X_train_mc)
X_val_mc_s   = scaler_mc.transform(X_val_mc)
X_test_mc_s  = scaler_mc.transform(X_test_mc)

# --- Responsible AI: check class balance ---
print('\nClass distribution (collision_severity):')
print(y_mc.value_counts(normalize=True).round(3))
# Note: if severely imbalanced, consider class_weight='balanced'

# Multiple algorithms on multiclass
mc_models = {
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Decision Tree':     DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
}

mc_results = {}
for name, model in mc_models.items():
    model.fit(X_train_mc_s, y_train_mc)
    val_acc  = model.score(X_val_mc_s, y_val_mc)
    val_f1   = f1_score(y_val_mc, model.predict(X_val_mc_s), average='weighted')
    mc_results[name] = {'val_accuracy': val_acc, 'val_f1': val_f1, 'model': model}
    print(f'{name:25s}  val acc: {val_acc:.3f}  val F1: {val_f1:.3f}')

# Best model on test set
best_mc_name = max(mc_results, key=lambda k: mc_results[k]['val_f1'])
best_mc = mc_results[best_mc_name]['model']
y_pred_mc = best_mc.predict(X_test_mc_s)

print(f'\nBest model: {best_mc_name}')
print(classification_report(y_test_mc, y_pred_mc))

cm = confusion_matrix(y_test_mc, y_pred_mc)
disp = ConfusionMatrixDisplay(cm, display_labels=best_mc.classes_)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_mc_name} (multiclass)')
plt.tight_layout()
plt.show()

# K-fold cross-validation on best model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_mc, X_mc, y_mc, cv=skf, scoring='f1_weighted')
print(f'\n5-fold CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

# -------------------------------------------------------
# TASK B: BINARY — urban_or_rural_area (Urban=1 / Rural=0)
# -------------------------------------------------------

binary_features = ['speed_limit', 'road_type', 'first_road_class',
                   'weather_conditions', 'light_conditions']

bin_df = df[binary_features + ['urban_or_rural_area']].dropna()

le_bin = LabelEncoder()
for col in bin_df.select_dtypes(include='object').columns:
    bin_df[col] = le_bin.fit_transform(bin_df[col].astype(str))

X_bin = bin_df[binary_features]
y_bin = bin_df['urban_or_rural_area']

X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

scaler_bin = StandardScaler()
X_tr_b_s = scaler_bin.fit_transform(X_tr_b)
X_te_b_s  = scaler_bin.transform(X_te_b)

# Hyperparameter tuning on Random Forest for binary task
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring='f1', n_jobs=-1)
rf_grid.fit(X_tr_b_s, y_tr_b)
print(f'\nBinary RF best params: {rf_grid.best_params_}')
print(f'Binary RF test F1: {f1_score(y_te_b, rf_grid.predict(X_te_b_s)):.3f}')
print(classification_report(y_te_b, rf_grid.predict(X_te_b_s)))

# -------------------------------------------------------
# TASK C: CATEGORICAL — junction_detail (7 classes)
# -------------------------------------------------------

cat_features = ['local_authority_district', 'road_type', 'speed_limit',
                'first_road_class', 'weather_conditions']

cat_df = df[cat_features + ['junction_detail']].dropna()

le_cat = LabelEncoder()
for col in cat_df.select_dtypes(include='object').columns:
    cat_df[col] = le_cat.fit_transform(cat_df[col].astype(str))

X_cat = cat_df[cat_features]
y_cat = cat_df['junction_detail']

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

scaler_cat = StandardScaler()
X_tr_c_s = scaler_cat.fit_transform(X_tr_c)
X_te_c_s  = scaler_cat.transform(X_te_c)

rf_cat = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_cat.fit(X_tr_c_s, y_tr_c)
print('\nCategorical (junction_detail) classification report:')
print(classification_report(y_te_c, rf_cat.predict(X_te_c_s)))

# =========================
# 5. SUPERVISED LEARNING

sheffield_dataframe = pd.read_csv('Sheffield Collision Data Cleaned.csv')

# SANITY CHECKING
print("")
print(sheffield_dataframe.shape)    #   Displaying the number of columns and rows within the dataset.
print(sheffield_dataframe.isnull().sum())   #   How many items within each column are null? - result shows none.
print("")

#       PAIR PLOT

#sbn.pairplot(sheffield_dataframe, plot_kws={'color': 'orange'})

#       CORRELATION HEATMAP

x = sheffield_dataframe.select_dtypes(include=["number"])

x = x.drop(columns=["collision_adjusted_severity_serious","collision_adjusted_severity_slight"],
    errors="ignore"
)   #Dropping the items present within the column that I will use as the label.

y =sheffield_dataframe['collision_adjusted_severity_serious'].map({"Not serious": 0, "Serious": 1}) #As an integer so it gives a numeric value
print(y.isna().sum())   #Ensuring none of the y values are null.

#Displaying a Correlation Heatmap (based on the x values entered above)
corr_matrix = x.corr()

plt.figure(figsize=(14, 14))
sbn.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

#       Strongly Correlated

#   collision_year & collision_injury_based
#   enhanced_severity_collision & collision_injury_based
#   attend_scene_of_accident & enhanced_severity_collision
#   collision_year & collision_injury_based

x = sheffield_dataframe[['collision_year']]
y = sheffield_dataframe[['collision_injury_based']]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=4)

linear_regression_model = LinearRegression()

linear_regression_model.fit(x_train, y_train)

predictions = x_test

plt.figure(figsize=(8, 6))

# Scatter plot: Actual vs Predicted values
plt.scatter(y_test, predictions, edgecolor='black', alpha=0.7, color='plum', label='Predicted Points')

# Regression line (best fit) through predicted vs actual values
z = np.polyfit(y_test, predictions, 1)  # Linear fit (degree=1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='red', linewidth=2, label='Regression Line')

# Perfect prediction line (y=x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='green', linewidth=2, label='Perfect Prediction')

# Labels and Title
plt.xlabel('Actual collision year (Y Test)', fontsize=12, weight='bold')
plt.ylabel('Predicted Collision Year (Y Pred)', fontsize=12, weight='bold')
plt.title('Actual vs Predicted House Prices with Regression Line', fontsize=14, weight='bold')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

#   PLOT COST FUNCTION

# --- Assume X_test, y_test, predictions exist ---
X = x_test.to_numpy().reshape(-1)
y = y_test.to_numpy().reshape(-1)

# --- Normalize X and y for stable plotting ---
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

# --- Assume predictions exist ---
pred = predictions.reshape(-1)
pred_norm = (pred - y_mean) / y_std

# --- Create small grid around predictions for visualization ---
# We'll simulate perturbing predictions slightly to see effect on MAE/MSE/RMSE
delta = 0.1  # small variation
theta0_vals = np.linspace(-delta, delta, 50)  # small intercept shift
theta1_vals = np.linspace(0.9, 1.1, 50)      # small slope multiplier
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

# --- Initialize metric surfaces ---
MAE_surface = np.zeros(T0.shape)
MSE_surface = np.zeros(T0.shape)
RMSE_surface = np.zeros(T0.shape)

# --- Compute metrics for perturbed predictions ---
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        y_pred = T0[i, j] + T1[i, j] * pred_norm  # perturb predictions
        MSE_surface[i, j] = metrics.mean_squared_error(y_norm, y_pred)
        MAE_surface[i, j] = metrics.mean_absolute_error(y_norm, y_pred)
        RMSE_surface[i, j] = np.sqrt(metrics.mean_squared_error(y_norm, y_pred))

# --- Plot 3D surfaces ---
fig = plt.figure(figsize=(22, 6))

# ---- MAE subplot ----
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(T0, T1, MAE_surface, cmap='plasma', alpha=0.9, edgecolor='k', linewidth=0.2)
ax1.set_xlabel("Theta0 Shift", labelpad=12)
ax1.set_ylabel("Theta1 Multiplier", labelpad=12)
ax1.set_zlabel("MAE", labelpad=15)
ax1.set_title("MAE Surface", pad=20)
ax1.view_init(elev=35, azim=135)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1).set_label('MAE')

# ---- MSE subplot ----
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(T0, T1, MSE_surface, cmap='viridis', alpha=0.9, edgecolor='k', linewidth=0.2)
ax2.set_xlabel("Theta0 Shift", labelpad=12)
ax2.set_ylabel("Theta1 Multiplier", labelpad=12)
ax2.set_zlabel("MSE", labelpad=15)
ax2.set_title("MSE Surface", pad=20)
ax2.view_init(elev=35, azim=135)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1).set_label('MSE')

# ---- RMSE subplot ----
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(T0, T1, RMSE_surface, cmap='cividis', alpha=0.9, edgecolor='k', linewidth=0.2)
ax3.set_xlabel("Theta0 Shift", labelpad=12)
ax3.set_ylabel("Theta1 Multiplier", labelpad=12)
ax3.set_zlabel("RMSE", labelpad=15)
ax3.set_title("RMSE Surface", pad=20)
ax3.view_init(elev=35, azim=135)
ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1).set_label('RMSE')

plt.tight_layout(w_pad=3)
plt.show()

#   Predicting based on user input

avg_income = float(input("Enter Avg. Area Income(in $): "))

# Create a DataFrame with the correct column name for prediction
input_features_df = pd.DataFrame({'Avg. Area Income': [avg_income]})

# Predict the house price
predicted_price = predict(input_features_df)

print(f"\n Predicted House Price: ${predicted_price[0]:,.2f}")

# =========================
# 6. REGRESSION

df_reg = pd.read_csv('Sheffield Collision Data Cleaned.csv')

reg_features = [
    'weather_conditions', 'light_conditions', 'road_surface_conditions',
    'junction_detail', 'junction_control', 'speed_limit',
    'urban_or_rural_area', 'day_of_week', 'hour'
]

reg_df = df_reg[reg_features + ['number_of_casualties']].dropna()

le_reg = LabelEncoder()
for col in reg_df.select_dtypes(include='object').columns:
    reg_df[col] = le_reg.fit_transform(reg_df[col].astype(str))

X_reg = reg_df[reg_features]
y_reg = reg_df['number_of_casualties']

# Train / val / test
X_tmp_r, X_te_r, y_tmp_r, y_te_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)
X_tr_r, X_vl_r, y_tr_r, y_vl_r = train_test_split(
    X_tmp_r, y_tmp_r, test_size=0.25, random_state=42)

scaler_reg = StandardScaler()
X_tr_r_s = scaler_reg.fit_transform(X_tr_r)
X_vl_r_s  = scaler_reg.transform(X_vl_r)
X_te_r_s  = scaler_reg.transform(X_te_r)

reg_models = {
    'Linear Regression':    LinearRegression(),
    'Ridge':                Ridge(alpha=1.0),
    'Lasso':                Lasso(alpha=0.1),
    'Random Forest Reg':    RandomForestRegressor(n_estimators=100, random_state=42),
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_tr_r_s, y_tr_r)
    preds = model.predict(X_vl_r_s)
    mae  = mean_absolute_error(y_vl_r, preds)
    rmse = mean_squared_error(y_vl_r, preds) ** 0.5
    r2   = r2_score(y_vl_r, preds)
    reg_results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'model': model}
    print(f'{name:22s}  MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}')

# Best on test set
best_reg_name = min(reg_results, key=lambda k: reg_results[k]['rmse'])
best_reg = reg_results[best_reg_name]['model']
y_pred_reg = best_reg.predict(X_te_r_s)

print(f'\nBest regression model: {best_reg_name}')
print(f'Test MAE:  {mean_absolute_error(y_te_r, y_pred_reg):.3f}')
print(f'Test RMSE: {mean_squared_error(y_te_r, y_pred_reg)**0.5:.3f}')
print(f'Test R²:   {r2_score(y_te_r, y_pred_reg):.3f}')

# Actual vs predicted plot (correctly labelled this time)
plt.figure(figsize=(8, 6))
plt.scatter(y_te_r, y_pred_reg, alpha=0.4, color='steelblue', edgecolors='none', s=15)
plt.plot([y_te_r.min(), y_te_r.max()],
         [y_te_r.min(), y_te_r.max()], 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual number of casualties')
plt.ylabel('Predicted number of casualties')
plt.title(f'Actual vs Predicted Casualties — {best_reg_name}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Trend analysis: collisions per year using groupby + regression
yearly = df_reg.groupby('collision_year').size().reset_index(name='collision_count')
X_yr = yearly[['collision_year']]
y_yr = yearly['collision_count']

lr_trend = LinearRegression().fit(X_yr, y_yr)
plt.figure(figsize=(9, 5))
plt.bar(yearly['collision_year'], yearly['collision_count'], color='steelblue', alpha=0.6, label='Actual')
plt.plot(yearly['collision_year'], lr_trend.predict(X_yr), 'r--', linewidth=2, label='Trend (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Number of collisions')
plt.title('Sheffield collision trend over time')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# =========================
# 7. UNSUPERVISED LEARNING

sheffield_df = pd.read_csv('Sheffield Collision Data Cleaned.csv')

# --- Feature selection & encoding ---
cluster_features = [
    'number_of_casualties', 'number_of_vehicles', 'speed_limit',
    'road_type', 'weather_conditions', 'light_conditions',
    'road_surface_conditions', 'urban_or_rural_area'
]

cluster_df = sheffield_df[cluster_features].copy().dropna()

# Encode any remaining categoricals
le = LabelEncoder()
for col in cluster_df.select_dtypes(include='object').columns:
    cluster_df[col] = le.fit_transform(cluster_df[col].astype(str))

scaler_unsup = StandardScaler()
X_cluster = scaler_unsup.fit_transform(cluster_df)

# --- Elbow Method to find optimal k ---
inertias = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o', color='steelblue')
plt.title('Elbow Method — Optimal Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# --- Silhouette scores to confirm optimal k ---
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    sil_scores.append(silhouette_score(X_cluster, labels))

plt.figure(figsize=(8, 5))
plt.plot(k_range, sil_scores, marker='o', color='darkorange')
plt.title('Silhouette Score by Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette score')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

best_k = k_range[sil_scores.index(max(sil_scores))]
print(f'Optimal k by silhouette: {best_k}  (score: {max(sil_scores):.3f})')

# --- Final KMeans with optimal k ---
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_cluster)
cluster_df['cluster'] = cluster_labels

print(f'\nKMeans Silhouette Score (k={best_k}): {silhouette_score(X_cluster, cluster_labels):.3f}')
print('\nCluster sizes:')
print(cluster_df['cluster'].value_counts())

# --- PCA to 2D for visualisation ---
pca_unsup = PCA(n_components=2)
X_2d = pca_unsup.fit_transform(X_cluster)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                      c=cluster_labels, cmap='tab10', s=15, alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title(f'KMeans Clusters (k={best_k}) — PCA projection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# --- Cluster profile — what conditions define each cluster? ---
cluster_df_orig = sheffield_df[cluster_features].copy().dropna()
cluster_df_orig['cluster'] = cluster_labels

print('\nCluster profiles (mean values):')
print(cluster_df_orig.groupby('cluster').mean().round(2).T)

# --- Algorithm 2: DBSCAN (for comparison) ---
# Responsible AI note: DBSCAN makes no assumption about cluster shape,
# useful for finding organic accident hotspot regions
dbscan = DBSCAN(eps=1.2, min_samples=10)
db_labels = dbscan.fit_predict(X_cluster)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
print(f'\nDBSCAN — clusters found: {n_clusters_db}, noise points: {n_noise}')

if n_clusters_db > 1:
    mask = db_labels != -1
    db_sil = silhouette_score(X_cluster[mask], db_labels[mask])
    print(f'DBSCAN Silhouette Score: {db_sil:.3f}')

plt.figure(figsize=(9, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1],
            c=db_labels, cmap='tab10', s=15, alpha=0.5)
plt.title('DBSCAN Clusters — PCA projection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 8. EVALUATION



print('=' * 60)
print('PERFORMANCE EVALUATION SUMMARY')
print('=' * 60)

# --- Classification comparison table ---
eval_rows = []
for name, res in mc_results.items():
    model = res['model']
    preds = model.predict(X_test_mc_s)
    eval_rows.append({
        'Model': name,
        'Task': 'Multiclass (severity)',
        'Accuracy': model.score(X_test_mc_s, y_test_mc),
        'F1 Weighted': f1_score(y_test_mc, preds, average='weighted'),
        'F1 Macro': f1_score(y_test_mc, preds, average='macro'),
    })

eval_df = pd.DataFrame(eval_rows)
print('\nClassification comparison:')
print(eval_df.to_string(index=False))

# Visual bar chart of F1 scores
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(eval_df))
ax.bar([i - 0.2 for i in x], eval_df['Accuracy'],   width=0.35, label='Accuracy',   color='steelblue')
ax.bar([i + 0.2 for i in x], eval_df['F1 Weighted'], width=0.35, label='F1 Weighted', color='darkorange')
ax.set_xticks(list(x))
ax.set_xticklabels(eval_df['Model'], rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_title('Model comparison — multiclass severity classification')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# --- Regression comparison table ---
reg_eval_rows = []
for name, res in reg_results.items():
    reg_eval_rows.append({
        'Model': name,
        'MAE': res['mae'],
        'RMSE': res['rmse'],
        'R²': res['r2'],
    })

reg_eval_df = pd.DataFrame(reg_eval_rows)
print('\nRegression comparison:')
print(reg_eval_df.to_string(index=False))

# --- Clustering evaluation ---
print(f'\nClustering (KMeans k={best_k}):')
print(f'  Silhouette Score: {silhouette_score(X_cluster, cluster_labels):.3f}')
print(f'  Inertia: {kmeans_final.inertia_:.1f}')
print('\n  Interpretation: silhouette > 0.5 = good separation')
print('  Clusters represent distinct accident condition profiles')

# --- Responsible AI: misclassification analysis ---
misclassified = X_test_mc.copy()
misclassified['actual'] = y_test_mc.values
misclassified['predicted'] = y_pred_mc
errors = misclassified[misclassified['actual'] != misclassified['predicted']]
print(f'\nMisclassification rate: {len(errors)/len(misclassified)*100:.1f}%')
print('Most common error pairs:')
print(errors.groupby(['actual','predicted']).size().sort_values(ascending=False).head(5))

# =========================
# 9. VISUALISATION / DASHBOARD
# =========================

#    THIS IS SET UP IN ANOTHER FILE SYSTEM.
