# Importing all of the required libraries for data manipulation, visualisation,
# machine learning, dimensionality reduction, and geospatial analysis.

#import warnings
#warnings.filterwarnings('ignore')

print("SCRIPT RUNNING")

def breakLine():
    print("")

import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd

# Data Loading and preprocessing
# Loading in the raw Sheffield-specific collision dataset.

print("=" * 70)
print("SHEFFIELD ROAD COLLISION ANALYSIS — Loading Data")
print("=" * 70)

sheffield_dataframe = pd.read_csv('Collision Data - Sheffield ONLY.csv')

print(f"Dataset Shape: {sheffield_dataframe.shape}")
print("\nFirst 5 rows:")
print(sheffield_dataframe.head())
print("\nData types:")
print(sheffield_dataframe.dtypes)
print(f"\nTotal columns: {len(sheffield_dataframe.columns)}")
print(f"Total records: {len(sheffield_dataframe)}")

# Data Preprocessing

# Strategy overview:
#   - Numerical columns  → mean imputation (preserves distribution)
#   - Categorical columns → mode imputation (most realistic value)
#   - Outlier detection   → IQR method for key numerical features
#   - Before/after visualisations for all cleaned columns
#
# Responsible AI note:
# Imputation decisions are documented with justifications to ensure
# transparency and reproducibility. Dropping rows was avoided to
# preserve the full dataset for imbalanced class learning.

print("\n" + "=" * 70)
print("3. DATA PREPROCESSING")
print("=" * 70)

# Initial Search for missing values across all of the columns within the dataset.

# Loading the updated dataset that has been initially processed
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Updated.csv')

# local_authority_highway_current — Categorical

# Here, the histogram only shows one value (E08000019).
# Therefore, I can safely impute all NA values within the local_authority_highway_current column with this value.
# This ensures that there isn't any bias introduced into the dataset by filling the NA values with incorrect or inaccurate values.

print("\n--- Cleaning: local_authority_highway_current ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sbn.histplot(sheffield_dataframe_updated['local_authority_highway_current'],
             ax=axes[0])
axes[0].set_title('local_authority_highway_current — BEFORE')

na_before = sheffield_dataframe_updated['local_authority_highway_current'].isna().sum()
print(f"  N/A before: {na_before}")

sheffield_dataframe_updated['local_authority_highway_current'] = (
    sheffield_dataframe_updated['local_authority_highway_current']
    .fillna('E08000019')
)

sbn.histplot(sheffield_dataframe_updated['local_authority_highway_current'],
             ax=axes[1])
axes[1].set_title('local_authority_highway_current — AFTER')
plt.tight_layout()
plt.show()

na_after = sheffield_dataframe_updated['local_authority_highway_current'].isna().sum()
print(f"  N/A after:  {na_after}")


# Latitude & Longitude - Numerical

# Here, I have filled the columns with the mean values.
# I have done this as both of the columns are continuous and the mean is a good way to ensure the data is kept as accurate as possible, 
# whilst ensuring that the coordinates do not go out of the Sheffield city boundaries.

# I have also noticed some outliers, I have purposely kept these as they are.
# These outliers are still accurate representations of real-world collisions, regardless of the location.

print("\n--- Cleaning: latitude & longitude ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sbn.histplot(sheffield_dataframe_updated['latitude'], bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Latitude — BEFORE')
sbn.histplot(sheffield_dataframe_updated['longitude'], bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Longitude — BEFORE')

for col in ['latitude', 'longitude']:
    mean_val = sheffield_dataframe_updated[col].mean()
    sheffield_dataframe_updated[col] = sheffield_dataframe_updated[col].fillna(mean_val)
    print(f"  {col}: filled with mean={mean_val:.4f}, "
          f"remaining NAs={sheffield_dataframe_updated[col].isna().sum()}")

sbn.histplot(sheffield_dataframe_updated['latitude'], bins=50, ax=axes[1, 0])
axes[1, 0].set_title('Latitude — AFTER')
sbn.histplot(sheffield_dataframe_updated['longitude'], bins=50, ax=axes[1, 1])
axes[1, 1].set_title('Longitude — AFTER')
plt.suptitle('Geographical Features — Before & After Imputation', fontsize=13)
plt.tight_layout()
plt.show()

# Easting & Northing — Numerical

# Here, I have done the same as I have above for Latitude and Longitude. 
# I have filled the columns with the mean values.

print("\n--- Cleaning: easting & northing ---")
for col in ['location_easting_osgr', 'location_northing_osgr']:
    mean_val = sheffield_dataframe_updated[col].mean()
    na_count = sheffield_dataframe_updated[col].isna().sum()
    sheffield_dataframe_updated[col] = sheffield_dataframe_updated[col].fillna(mean_val)
    print(f"  {col}: filled {na_count} NAs with mean={mean_val:.1f}")

# collision_adjusted_severity_serious & collision_severity_slight — Binary

# Here, I have used mode imputation as it is the most realistic for binary values.

print("\n--- Cleaning: collision_adjusted_severity_serious & _slight ---")
for col in ['collision_adjusted_severity_serious',
            'collision_adjusted_severity_slight']:
    mode_val = sheffield_dataframe_updated[col].mode()[0]
    na_count = sheffield_dataframe_updated[col].isna().sum()
    sheffield_dataframe_updated[col] = (
        sheffield_dataframe_updated[col].fillna(mode_val)
    )
    print(f"  {col}: filled {na_count} NAs with mode={mode_val}")

# Map binary integer flags to readable labels
sheffield_dataframe_updated["collision_adjusted_severity_serious"] = (
    sheffield_dataframe_updated["collision_adjusted_severity_serious"]
    .astype(int).map({0: "Not serious", 1: "Serious"})
)
sheffield_dataframe_updated["collision_adjusted_severity_slight"] = (
    sheffield_dataframe_updated["collision_adjusted_severity_slight"]
    .astype(int).map({0: "Not slight", 1: "Slight"})
)

# IQR-based outlier detection
# This is used for key numerical features to identify any potential issues with the quality of the data and understand the distribution of the values.

# Here, the outliers are flagged and reported but not removed.
# Similar to my previous point in the latitude and longitude section, these outliers are important as they are still accurate representations of real-world events.
# Therefore, removing them would introduce selection bias into the dataset and reduce the accuracy of the model.

print("\n--- Outlier Detection (IQR method) ---")
outlier_cols = ['number_of_casualties', 'number_of_vehicles', 'speed_limit']

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(16, 5))
for i, col in enumerate(outlier_cols):
    if col in sheffield_dataframe_updated.columns:
        col_data = sheffield_dataframe_updated[col].dropna()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((col_data < lower) | (col_data > upper)).sum()
        print(f"  {col}: Q1={Q1:.1f}, Q3={Q3:.1f}, "
              f"bounds=[{lower:.1f}, {upper:.1f}], outliers={n_outliers} "
              f"({100*n_outliers/len(col_data):.1f}%) — RETAINED")
        sbn.boxplot(y=col_data, ax=axes[i], color='skyblue',
                    flierprops=dict(marker='o', markerfacecolor=(0.7, 0.2, 0.4),
                                    markersize=4, alpha=0.6))
        axes[i].set_title(f'{col}\n(outliers shown in pink)')

plt.suptitle('Outlier Analysis — Key Numerical Features', fontsize=13)
plt.tight_layout()
plt.show()

# Final check for any remaining missing values after cleaning
# This is just to ensure that any NA values have been handled correctly.
# There shouldn't be any N/A values within the dataset at this point.

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

























print("Message 1")
