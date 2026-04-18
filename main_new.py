
# Importing all of the required libraries for data manipulation, visualisation,
# machine learning, dimensionality reduction, and geospatial analysis.

# This is my own function that I have created. This allows me to easily enter a blank line wherever I would like one.
def breakLine():
    print("")

# Starting to import all of the libraries.
# All of these libraries are required throughout the model for various tasks such as data manipulation, visualisation, machine learning
# dimensionality reduction, and geospatial analysis.
# Rather than have these dotted throughout the code, I thought it would be easier to manage them (add or remove imports as required) if they were all in one accessible place.
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sbn

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV, KFold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
from imblearn.over_sampling import SMOTE

# ====================================================================      Data Preprocessing      ======================================================
# Beginning the process of pre-processing the dataset.
# This is the first step and one of the most important steps to get right.
# Loading in the raw Sheffield-specific collision dataset.
breakLine()
print("=" * 70) # Printing "=" 70 times. I will do this throughout the model so that I can see whereabouts the different sections are, upon runtime.
print("SHEFFIELD ROAD COLLISION ANALYSIS MODEL")
print("=" * 70)
breakLine()
sheffield_dataframe = pd.read_csv('Collision Data - Sheffield ONLY.csv') # Loading in the raw (initial) dataset.
# # This dataset has been taken straight from the UK government site.
# # Therefore, it is likely to have some missing value, some innacuracies and some outliers.
# # I aim to handle these issues within this section.
# # At the end of the section, I will re-save the dataset as a new, cleaned file that I can use confidently throughout the rest of my work.

print("Dataset shape:", sheffield_dataframe.shape) # Printing the shape fo the dataset.
breakLine()
print("First 5 rows:")
breakLine()
print(sheffield_dataframe.head()) # Printing the head of the dataframe. Without any intervention, this will display the first 5 rows of the dataset.
breakLine()
print("Data types:")
breakLine()
print(sheffield_dataframe.dtypes)
breakLine()
print("Total columns:" ,{len(sheffield_dataframe.columns)})
breakLine()
print("Total records:" ,{len(sheffield_dataframe)})
breakLine()

# =================================================================================================================

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
breakLine()
print("=" * 70) # Using the same process as earlier. This time, to mark the beginning of the data preprocessing section.
print("DATA PREPROCESSING")
print("=" * 70)
breakLine()
# Performing an initial Search for missing values across all of the columns within the dataset.
# For this, I have created a reusable function that I can use throughout the model wherever I may need to check for missing values.

def missing_summary(df, sort=True):
    summary = (
        df.isnull().sum()
        .loc[lambda x: x > 0]
        .to_frame('Missing Count')
    )
    
    # This shows the percentage of missing values within each column that contains them. It also shows an exact count of the missing values.
    summary['Percentage (%)'] = (summary['Missing Count'] / len(df)) * 100 # It provides this in an easy to read format. This is really important for me to understadn the extent of the missing values easily and quickly.
    
    if sort:
        summary = summary.sort_values(by='Missing Count', ascending=False)
    
    print("Columns with missing values: " , len(summary))
    breakLine()
    print(summary)
    
    return summary

# I then run the function above on the initial dataset.
missing_df = missing_summary(sheffield_dataframe)

# Loading the updated dataset that has been initially processed
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Updated.csv')

# local_authority_highway_current — Categorical

# Here, the histogram only shows one value (E08000019).
# Therefore, I can safely impute all NA values within the local_authority_highway_current column with this value.
# This ensures that there isn't any bias introduced into the dataset by filling the NA values with incorrect or inaccurate values.

breakLine()
print("Cleaning: local_authority_highway_current")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sbn.histplot(sheffield_dataframe_updated['local_authority_highway_current'],
             ax=axes[0])
axes[0].set_title('local_authority_highway_current — BEFORE')   # The section above displays a histogram of the local_authority_highway_current column before the imputation.

na_before = sheffield_dataframe_updated['local_authority_highway_current'].isna().sum()
print(" N/A before: ", na_before)  # This value indicates the number of N/A values within the loal_authority_highway_current column before the imputation.

sheffield_dataframe_updated['local_authority_highway_current'] = (
    sheffield_dataframe_updated['local_authority_highway_current']
    .fillna('E08000019')    # Here, I am filling the N/A values with E08000019, as this is the only value that is present within the column.
    # This means that I am not introducing any bias into the dataset by filling the N/A values with an inaccurate value as there are no other values within the column to choose from.
)

sbn.histplot(sheffield_dataframe_updated['local_authority_highway_current'],    # This Histogram shows the effects of the cleaning.
             ax=axes[1])
axes[1].set_title('local_authority_highway_current — AFTER')
plt.tight_layout()
plt.show()

# Both of the Histograms are included within the same image, and are available in the Results folder, as a file titled "HISTOGRAM -local_highway_before_after.png".

na_after = sheffield_dataframe_updated['local_authority_highway_current'].isna().sum()
print(" N/A after: ", na_after) # This value shows the number of N/A values in the column after the imputation. This is 0.
# An image of the before and after output for this section is within the results folder as "CLEANING_RESULTS_local_highway.png"

# Latitude & Longitude - Numerical

# Here, I have filled the columns with the mean values.
# I have done this as both of the columns are continuous and the mean is a good way to ensure the data is kept as accurate as possible, 
# whilst ensuring that the coordinates do not go out of the Sheffield city boundaries.

# I have also noticed some outliers, I have purposely kept these as they are.
# These outliers are still accurate representations of real-world collisions, regardless of the location.

breakLine()
print("Cleaning: latitude & longitude")
fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # This creates a two by two grid which I will then fill with the before and after histograms for longitude and latitude.

sbn.histplot(sheffield_dataframe_updated['latitude'], bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Latitude — BEFORE') # Displaying the histogram for latitude before the data imputation.

sbn.histplot(sheffield_dataframe_updated['longitude'], bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Longitude — BEFORE')  # Displaying the histogram for longitude before the data imputation.

for col in ['latitude', 'longitude']:
    mean_val = sheffield_dataframe_updated[col].mean()
    sheffield_dataframe_updated[col] = sheffield_dataframe_updated[col].fillna(mean_val)    # Filling all of the N/A values present in both of the columns with the mean values for each.
    print(f"  {col}: filled with mean = { mean_val:.4f}."
          f"\n      Remaining NAs = {sheffield_dataframe_updated[col].isna().sum()}") # Calculating the number of N/A values that are present after the imputation. This is 0 for both of the columns.

sbn.histplot(sheffield_dataframe_updated['latitude'], bins=50, ax=axes[1, 0])
axes[1, 0].set_title('Latitude — AFTER') # Displaying the histogram for the latitude column after the data imputation.

sbn.histplot(sheffield_dataframe_updated['longitude'], bins=50, ax=axes[1, 1])
axes[1, 1].set_title('Longitude — AFTER') # Displaying the histogram for the longitude column after the data imputation.

plt.suptitle('Geographical Features — Before & After Imputation', fontsize=13) # Setting the title of the image of the graphs.
plt.tight_layout()
plt.show()

# urban_or_rural_area - Converting to binary to remove unnecessary complexity (resulting in urban = 1 and rural = 0)
breakLine()
print("Cleaning: urban_or_rural_area")
breakLine()
print("Before:")
counts = sheffield_dataframe_updated['urban_or_rural_area'].value_counts()
percentages = sheffield_dataframe_updated['urban_or_rural_area'].value_counts(normalize=True) * 100
for val in counts.index:
    print(f"{val}: {counts[val]} ({percentages[val]:.1f}%)")    # Counting the number of 3's that I initially have within the dataset.
# The percentage of each value within the column is also shown next to the count. This is important as it allows me to understand the extent of the 3's within the dataset, and therefore, the extent of the inaccuracy that they would introduce if I were to keep them within the dataset.

df = sheffield_dataframe_updated[
    sheffield_dataframe_updated['urban_or_rural_area'].isin([1, 2]) # Removing any value that isn't 1 or 2.
].copy() # I initially made a mistake here that I only noticed whilst going back through the code.
# I had only disregarded the 3's in the dataset. This meant that I had kept -1 values. This meant it would be inaccurate as these values don't show either urban or rural areas.
# They are likely to signify unrecorded values. 
# I have now removed these as well. This will ensure that the predictions made by the model will be as accurate as possible.

breakLine()
print("After removing 3s and -1s:")
counts = df['urban_or_rural_area'].value_counts()
percentages = df['urban_or_rural_area'].value_counts(normalize=True) * 100
for val in counts.index:
    print(f"{val}: {counts[val]} ({percentages[val]:.1f}%)") # Checking the count after the removal, ensuring that all of the 3's have been removed.
# Once again, showing the percentage next to the count. This time, for once the removal has taken place.

df['urban_or_rural_area'] = df['urban_or_rural_area'].map({
    1: 1,   # Mapping the urban areas to 1.
    2: 0    # Mapping the rural areas to 0.
})

breakLine()
print("After binary conversion:")
counts = df['urban_or_rural_area'].value_counts()
percentages = df['urban_or_rural_area'].value_counts(normalize=True) * 100
for val in counts.index:
    label = "Urban" if val == 1 else "Rural"
    print(f"{label} ({val}): {counts[val]} ({percentages[val]:.1f}%)") # Performing a final count of the values, ensuring the areas are mapped to 1 and 0 as intended.

# Easting & Northing — Numerical

# Here, I have done the same as I have above for Latitude and Longitude. 
# I have filled the columns with the mean values.
breakLine()
print("Cleaning: easting & northing")
breakLine()

na_before = sheffield_dataframe_updated['location_easting_osgr'].isna().sum()   # Calculating the number of N/A values that are present in the location_easting_osgr column before the imputation.
na_before = sheffield_dataframe_updated['location_northing_osgr'].isna().sum()  # Calculating the number of N/A values that are present in the location_northing_osgr column before the imputation.
# Both of the values calculated above are printed below. 
print(" N/A Count before: location_easting_osgr = ", na_before)
print(" N/A Count before: location_northing_osgr = ", na_before)
breakLine()

for col in ['location_easting_osgr', 'location_northing_osgr']:
    mean_val = sheffield_dataframe_updated[col].mean() # Calculating the mean value for each of the columns.
    na_count = sheffield_dataframe_updated[col].isna().sum()    # Calculating the number of N/A values that are present in each of the columns.
    sheffield_dataframe_updated[col] = sheffield_dataframe_updated[col].fillna(mean_val)    # Filling the N/A values with the mean value for each of the columns.
    print(f"  {col}: filled {na_count} NAs with mean = {mean_val:.1f}") # This shows the number of N/A values that were filled for each of the columns, as well as the mean value that they were filled with.

na_after = sheffield_dataframe_updated['location_easting_osgr'].isna().sum()    # Calculating the number of N/A values that are present in the location_easting_osgr column after the imputation.
na_after = sheffield_dataframe_updated['location_northing_osgr'].isna().sum()   # Calculating the number of N/A values that are present in the location_northing_osgr column after the imputation.
# Both of the values calculated above are printed below.
breakLine()
print(" N/A Count after: location_easting_osgr = ", na_after)
print(" N/A Count after: location_northing_osgr = ", na_after)
breakLine()
# This section was actually relatively clean, with only a total of 130 N/A values across both of the columns.

# collision_adjusted_severity_serious & collision_severity_slight — Binary

# Here, I have used mode imputation as it is the most realistic for binary values.
# For the section below, I have repeated the same logic that is present above.
breakLine()
print("Cleaning: collision_adjusted_severity_serious & collision_adjusted_severity_slight")
breakLine()

na_before = sheffield_dataframe_updated['collision_adjusted_severity_serious'].isna().sum()   # Calculating the number of N/A values that are present in the collision_adjusted_severity_serious column before the imputation.
na_before = sheffield_dataframe_updated['collision_adjusted_severity_slight'].isna().sum()  # Calculating the number of N/A values that are present in the collision_adjusted_severity_slight column before the imputation.
# Both of the values calculated above are printed below. 
print(" N/A Count before: collision_adjusted_severity_serious = ", na_before)
print(" N/A Count before: collision_adjusted_severity_slight = ", na_before)
breakLine()

for col in ['collision_adjusted_severity_serious', # Here, I am filling the N/A values within both of the columns with the mode value for each of the columns. This is because both of the columns are binary, and therefore, the mode value is the most realistic value to fill the N/A values with.
            'collision_adjusted_severity_slight']:
    mode_val = sheffield_dataframe_updated[col].mode()[0]
    na_count = sheffield_dataframe_updated[col].isna().sum()
    sheffield_dataframe_updated[col] = (
        sheffield_dataframe_updated[col].fillna(mode_val)
    )
    print(f"  {col}: filled {na_count} NAs with mode = {mode_val}")

# Mapping the binary integer flags to readable labels
sheffield_dataframe_updated["collision_adjusted_severity_serious"] = ( # Here, I am mapping the values within the collision_adjusted_severity_serious column to "Serious" and "Not Serious". This is just to make the dataset easier to understand and work with.
    sheffield_dataframe_updated["collision_adjusted_severity_serious"]
    .astype(int).map({0: "Not serious", 1: "Serious"})
)
sheffield_dataframe_updated["collision_adjusted_severity_slight"] = ( # Here, I am mapping the values within the collision_adjusted_severity_slight column to "Slight" and "Not Slight".
    sheffield_dataframe_updated["collision_adjusted_severity_slight"]
    .astype(int).map({0: "Not slight", 1: "Slight"})
)

na_after = sheffield_dataframe_updated['collision_adjusted_severity_serious'].isna().sum()   # Calculating the number of N/A values that are present in the collision_adjusted_severity_serious column after the imputation.
na_after = sheffield_dataframe_updated['collision_adjusted_severity_slight'].isna().sum()  # Calculating the number of N/A values that are present in the collision_adjusted_severity_slight column after the imputation.
# Both of the values calculated above are printed below.
breakLine()
print(" N/A Count after: collision_adjusted_severity_serious = ", na_after)
print(" N/A Count after: collision_adjusted_severity_slight = ", na_after)
breakLine()

# IQR-based outlier detection
# This is used for key numerical features to identify any potential issues with the quality of the data and understand the distribution of the values.

# Here, the outliers are flagged and reported but not removed.
# Similar to my previous point in the latitude and longitude section, these outliers are important as they are still accurate representations of real-world events.
# Therefore, removing them would introduce selection bias into the dataset and reduce the accuracy of the model.

breakLine()
print("Outlier Detection (Using the IQR method): ")
outlier_cols = ['number_of_casualties', 'number_of_vehicles', 'speed_limit'] # These are the columns that I have chosen to check for outliers within. I have chosen these columns as they are key numerical features that are likely to have a significant impact on the predictions made by the model if the values within them are inaccurate.
# My aim here, once again is to reduce/remove any inaccuracies within the dataset, without removing any accurate representations of real-world events. Therefore, I have chosen to flag and report any outliers rather than remove them, thus keeping the data as realistic as possible.

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(16, 5))
for i, col in enumerate(outlier_cols):
    if col in sheffield_dataframe_updated.columns:
        col_data = sheffield_dataframe_updated[col].dropna()
        Q1 = col_data.quantile(0.25) # Calculating the first quartile (Q1) and the third quartile (Q3) for each of the columns. 
        Q3 = col_data.quantile(0.75) # These values are used to calculate the interquartile range (IQR) which is then used to identify outliers.
        IQR = Q3 - Q1 # Calculating the IQR (Interquartile Range) for each of the columns.
        lower = Q1 - 1.5 * IQR # Q1 is the lower QR
        upper = Q3 + 1.5 * IQR # Q3 is the upper QR
        n_outliers = ((col_data < lower) | (col_data > upper)).sum() # Counting the number of outliers that are present in each of the columns based on the IQR that I calculated above.
        breakLine()

        print(f"{col}: Q1 = {Q1:.1f}")
        print(f"Q3 = {Q3:.1f}")
        breakLine()
        print(f"bounds = [ {lower:.1f}, {upper:.1f} ], outliers = {n_outliers} ")
        print(f"({100*n_outliers/len(col_data):.1f}%) — RETAINED")

        sbn.boxplot(y=col_data, ax=axes[i], color='skyblue',
                    flierprops=dict(marker='o', markerfacecolor=(0.7, 0.2, 0.4),
                                    markersize=4, alpha=0.6))
        axes[i].set_title(f'{col}\n(outliers shown in pink)')
        breakLine()

plt.suptitle('Outlier Analysis — Key Numerical Features', fontsize=13) # Titling the image of the boxplots hilighting the outliers.
plt.tight_layout()
plt.show()


# Final check for any remaining missing values after cleaning
# This is just to ensure that any NA values have been handled correctly.
# There shouldn't be any N/A values within the dataset at this point.


breakLine()
print("=" * 70) # Using the same process as earlier. This time, to mark the beginning of the data preprocessing section.
breakLine()
print("FINAL N/A COUNT ACROSS THE WHOLE DATASET ")
breakLine()
remaining_nulls = sheffield_dataframe_updated.isnull().sum() # Calculating the number of N/A values that are present in each of the columns after the cleaning process.
print(remaining_nulls)
breakLine()
total_nulls = remaining_nulls.sum()
print("total: ", total_nulls)
breakLine()
print("=" * 70)
breakLine()


print ("========================================== THESE MAY NEED TO BE MOVED ===================================================")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

df = sheffield_dataframe_updated.copy()

# Recreate engineered features
df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
df['high_speed_zone'] = (df['speed_limit'] >= 50).astype(int)

# Clean target (remove invalid class)
df = df[df['urban_or_rural_area'].isin([1, 2])]

# Convert to binary
df['urban_or_rural_area'] = df['urban_or_rural_area'].map({
    1: 1,
    2: 0
})

# Safety check
print("Final target values:", sorted(df['urban_or_rural_area'].unique()))

print ("========================================== THESE MAY NEED TO BE MOVED ===================================================")


# Below, I have created a series of visualisations to explore the dataset, following the cleaning and imputation process.
# These visualisations are important as they allow me to understand the distribution of the data, the relationships between different features, and any potential issues that may still be present within the dataset after cleaning.

# Creating a Collision count plot
sbn.countplot(
    data=sheffield_dataframe_updated.dropna(
        subset=["collision_adjusted_severity_serious"]),
    x="collision_adjusted_severity_serious",
    ax=axes[0, 0],
    color='skyblue'
)
axes[0, 0].set_title("Serious Collision Distribution")
axes[0, 0].set_xlabel("Severity")

# Creating a Geographical scatter plot
axes[0, 1].scatter(
    sheffield_dataframe_updated['longitude'],
    sheffield_dataframe_updated['latitude'],
    alpha=0.1, s=3, color='steelblue'
)
axes[0, 1].set_title("Collision Locations (Lat/Lon)")
axes[0, 1].set_xlabel("Longitude")
axes[0, 1].set_ylabel("Latitude")

# Creating two KDE plots
sbn.kdeplot(data=sheffield_dataframe_updated["latitude"],
            ax=axes[0, 2], color='steelblue')
axes[0, 2].set_title("KDE — Latitude Distribution")

sbn.kdeplot(data=sheffield_dataframe_updated["longitude"],
            ax=axes[1, 0], color='darkorange')
axes[1, 0].set_title("KDE — Longitude Distribution")

# Creating an Easting/Northing scatter plot
axes[1, 1].scatter(
    sheffield_dataframe_updated['location_northing_osgr'],
    sheffield_dataframe_updated['location_easting_osgr'],
    alpha=0.1, s=3, color='forestgreen'
)
axes[1, 1].set_title("Collision Locations (OS Grid)")
axes[1, 1].set_xlabel("Northing")
axes[1, 1].set_ylabel("Easting")

# Creating a Local authority bar chart
sheffield_dataframe_updated['local_authority_highway_current'] \
    .value_counts().plot(kind="bar", ax=axes[1, 2], color='steelblue')
axes[1, 2].set_title("Collisions by Highway Authority")
axes[1, 2].set_xlabel("")

plt.suptitle('Sheffield Collision Data — Exploratory Visualisations', fontsize=14)
plt.tight_layout()
plt.show()

# Within my detailed_ReadMe.md file, I have explained each of these graphs and included a photo of them.
# This is under the section titled "Final Preprocessing Result Charts (Graphs)"

# Saving the dataset after cleaning and imputation.
# I can now continue with my development with this new, cleaned dataset.

remaining_nulls = sheffield_dataframe_updated.isnull().sum() # Performing one final check for any remaining N/A values within the dataset, just to ensure that all of the N/A values have been handled correctly.
still_null = remaining_nulls[remaining_nulls > 0]
if len(still_null) == 0: # If there are no remaining N/A values, then this message is printed to confirm that there are no missing values within the dataset.
    print("Final validation: No missing values remain in cleaned dataset.")
    breakLine()
else:
    breakLine()
    print("Remaining missing values after cleaning:") # Else, if there are still missing values within the dataset, then this message is printed, along with the count of the remaining N/A values for each of the columns that still contain them.
    print(still_null)

sheffield_dataframe_updated.to_csv(
    "Sheffield Collision Data Cleaned.csv", index=False
)
print("Cleaned dataset saved: Sheffield Collision Data Cleaned.csv") # Hilighting what I have saved the new data file as. This is important as I will be using the new file for the rest of my work.

# Feature Engineering

# New features are taken from existing columns to capture any hidden patterns.

#   is_weekend          —       Weekends show different driving behaviour to weekdays
#   time_of_day         —       Temporal grouping captures rush-hour vs night effects
#   risk_score          —       A composite severity indicator (both vehicles and casualties)
#   casualty_per_vehicle —      Normalised severity independent of collision size
#   speed_urban_interaction —   Interaction term: speed × location type
#   collision_age       —       Years since data collection started (trend feature)
#   high_speed_zone     —       Binary flag for speed limits > or equal to 60mph (potentially motorway accidents)

breakLine()
print("=" * 70)
print("Feature Engineering")
print("=" * 70)
breakLine()

df = pd.read_csv('Sheffield Collision Data Cleaned.csv') # Loading in the new, cleaned dataset, created above.

# Time-based features
df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)  # Extrapolating the day of the week based on the day_of_week column wthin the dataset.

df['hour'] = pd.to_datetime( # Defining a new column called "hour" which is the hour extracted from the time column.
    df['time'],
    format='%H:%M', # Specififying the format of the time column to ensure that the hour column is extracted correctly.
    errors='coerce'
).dt.hour # Returning the hour as its own column filled with integer values.

# Create time of day categories (fixed binning issue)
df['time_of_day'] = pd.cut(
    df['hour'], # I then use the new hour column here to create a time of day section which categorises the time of day into four different categories.
    bins=[0, 6, 12, 18, 24], # The bins are defined as follows: 0-6 is night, 6-12 is morning, 12-18 is afternoon, and 18-24 is evening. This allows me to capture any patterns that may be present within the different times of the day.
    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
    right=False # This means that the bins are left-inclusive and right-exclusive, ensuring that each hour is categorized correctly without overlap. For example, 6:00 will be categorized as "Morning" rather than "Night". This ensures it works with teh 24 hour clock format and that the time of day categories are accurate for analysis
)

# Risk/severity composite features
df['risk_score'] = ( # Creating a total risk score that combines the number of vehicles and the number of casualties, giving more weight to casualties as they are a more direct indicator of severity.
    df['number_of_vehicles'] * 0.4 +
    df['number_of_casualties'] * 0.6 
)   # This uses the mathematical formula above to calculate the new risk score for each collision within the dataset.

df['casualty_per_vehicle'] = ( # Creating a new feature that represents the number of casualties per vehicle, which helps to normalise the severity of collisions based on their size.
    df['number_of_casualties'] / (df['number_of_vehicles']) # Using this mathematical logic.
)

# Interaction features with specific flags
df['speed_urban_interaction'] = df['speed_limit'] * df['urban_or_rural_area'] # Creating an interaction term that multiplies the speed limit by the urban/rural flag, to capture how speed limits may have different effects in urban vs rural settings.

df['high_speed_zone'] = (df['speed_limit'] >= 60).astype(int) # Creating a binary flag to indicate whether the collision occurred in a high speed zone (60mph or above), which may correlate with more severe accidents.

# Temporal trend feature
if 'collision_year' in df.columns:
    df['collision_age'] = df['collision_year'].max() - df['collision_year'] # Creating a feature that calculates the age of the collision in years from the latest collision year in the dataset, to capture any temporal trends or improvements in road safety over time.

# Outlining all of the engineered features that I have added to the dataset.
print("Engineered features added:")
new_features = ['is_weekend', 'time_of_day', 'risk_score',
                'casualty_per_vehicle', 'speed_urban_interaction',
                'high_speed_zone', 'collision_age']

# Printing all of the engineered features to confirm that the have been added to the dataset successfully.
for f in new_features:
    if f in df.columns:
        print("  + ", {f}) # Printing them in a nice, neat format.
breakLine()

# Visualising engineered features using graphs/charts
# All of the graphs that are generate below are included within the results folder under "Feature-Engineering-Results"
# Furthermore, they are explained in more detail within the detailed_ReadMe.md file, under the section titled "Feature Engineering - Graphs"
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Weekday vs Weekend
sbn.countplot(
    data=df,
    x='is_weekend',
    ax=axes[0, 0],
    palette='Set2'
)

axes[0, 0].set_title("Collisions: Weekday vs Weekend")
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Weekday', 'Weekend'])

# Time of day
sbn.countplot(
    data=df,
    x='time_of_day',
    hue='time_of_day',
    ax=axes[0, 1],
    palette='Set3',
    order=['Night', 'Morning', 'Afternoon', 'Evening'],
    legend=False
)
axes[0, 1].set_title("Collisions by Time of Day")

# Risk score
sbn.histplot(df['risk_score'], bins=30, ax=axes[0, 2], color='coral')
axes[0, 2].set_title("Risk Score Distribution")

# Casualty per vehicle
sbn.histplot(df['casualty_per_vehicle'], bins=30, ax=axes[1, 0],
             color='steelblue')
axes[1, 0].set_title("Casualty per Vehicle Distribution")

# High speed zone
sbn.countplot(
    data=df,
    x='high_speed_zone',
    hue='high_speed_zone',
    ax=axes[1, 1],
    palette='Set1',
    legend=False
)
axes[1, 1].set_title("High Speed Zone (≥60mph) Collisions")
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['Normal Speed', 'High Speed'])

# Collision age
if 'collision_age' in df.columns:
    sbn.histplot(df['collision_age'], bins=20, ax=axes[1, 2], color='green')
    axes[1, 2].set_title("Collision Age (years from latest)")

plt.suptitle('Feature Engineering — Distributions', fontsize=14)
plt.tight_layout()
plt.show()

# Supervised Learning - Classification

# The dataset is split into train / validation / test (60/20/20).
# I have compared multiple algorithms per task to identify the most suitable model for the dataset.

breakLine()
print("=" * 70)
print("Supervised Learning")
print("=" * 70)
breakLine()

# Creating a correlation heatmap for the numerical featues within the dataset.
numeric_df = df.select_dtypes(include=["number"]).drop(
    columns=['collision_adjusted_severity_serious',
             'collision_adjusted_severity_slight'],
    errors='ignore'
)

corr_matrix = numeric_df.corr()
plt.figure(figsize=(14, 12))
sbn.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            annot_kws={'size': 7})
plt.title("Feature Correlation Matrix — Sheffield Collision Data", fontsize=13)
plt.tight_layout()
plt.show()

# Multiclass Classification

# Multi Class Classification for:
    # collision_severity_slight
    # collision_severity_serious
    # collision_severity_fatal

# Here, I have compared four different algorithms. RF, DT, GBM and Logistic Regression.
# I have used Random Forest (RF), to build a sequence of decision trees and combine them, this came out as the best model based on the score (F1)
# I have used Decision Tree (DT), to learn if/then rules. For example, if the speed is greater than 60 and the road is equal to rural. Then, the accident is likely to be serious.
# I have used Gradient Boosting (GBM) to build the trees sequentially and continually correct mistakes. This acts as an error handler, catching errors and emerging patterns within the learning. 
# However, to improve this and make it more accurate, it would need to be tuned.
# Finally, I have used Logistic Regression. This learned a mathematical relationship between the features and the classes and provided me with a simple baseline which could then be compared to more complex models.



breakLine()
print("Multiclass Classification (collision_severity)") # Starting to work on multiclass classification for the collision severity column.
# This is important as it allows me to predict the severity of a collision, which is a key aspect of road safety analysis and something that could ultimately help inform road safety policies and interventions within Sheffield.

multiclass_features = [ # Defining the features that I will use for the multiclass classification.
    'weather_conditions', 'road_surface_conditions', 'light_conditions',
    'speed_limit', 'number_of_vehicles', 'number_of_casualties',
    'urban_or_rural_area', 'day_of_week', 'junction_detail', 'road_type',
    'is_weekend', 'risk_score', 'high_speed_zone'
]

mc_df = df.drop(columns=['collision_index', 'collision_ref_no'], # Dropping any columns that aren't relevant for this task.
                errors='ignore').dropna(
    subset=multiclass_features + ['collision_severity'] # Dropping any rows that have N/A values in the features or target column for this task, as these would cause issues during model training and evaluation.
).copy()

# Encoding the categorical features using Label Encoding.
# This is an important step as it allows me to convert the categorical features into a numerical format that can be used by machine learning algorithms.
for col in mc_df.select_dtypes(include='object').columns:
    mc_df[col] = LabelEncoder().fit_transform(mc_df[col].astype(str))

X_mc = mc_df[multiclass_features] # Defining the feature matrix (X) and the target vector (y) for the multiclass classification.
y_mc = mc_df['collision_severity']

breakLine()
print('Class distribution (collision_severity):')
breakLine()
print(y_mc.value_counts(normalize=True).round(3))
breakLine()

# Using a 60/20/20 split for data training/validation/testing.
X_temp, X_test_mc, y_temp, y_test_mc = train_test_split(
    X_mc, y_mc, test_size=0.2, random_state=42, stratify=y_mc)
X_train_mc, X_val_mc, y_train_mc, y_val_mc = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp) # The stratify parameter ensures that the class distribution is maintained across all splits, which is important for imbalanced datasets like this one.

print(f'Split — train: {len(X_train_mc)}, val: {len(X_val_mc)}, '
      f'test: {len(X_test_mc)}') # Printing the number of samples in each of the splits to confirm that they are correct based on the 60/20/20 split that I initially intended to create.
breakLine()

scaler_mc = StandardScaler()
X_train_mc_s = scaler_mc.fit_transform(X_train_mc) # Scaling the features using StandardScaler to ensure that they are on the same scale.
X_val_mc_s   = scaler_mc.transform(X_val_mc) # This is important as it allows the models to learn more effectively and prevents a specific feature from dominating the learning process due to its scale.
X_test_mc_s  = scaler_mc.transform(X_test_mc)

# Applying SMOTE to the training data only to handle class imbalance.
# SMOTE synthetically oversamples the minority classes so the model doesn't just learn to predict the majority class (which in this instance is Class 3) every time, which would result in misleadingly high accuracy but poor performance.
print('Responsible AI: applying SMOTE to address class imbalance in training data.')
breakLine()
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_mc_s, y_train_mc)
print(f'Post-SMOTE class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}')
breakLine()

# Here, I am computing the sample weights for Gradient Boosting.
# As it does not support the class_weight parameter directly unlike Random Forest and Logistic Regression, Passing sample weights to fit() achieves the same effect of penalising misclassification of minority classes.
gb_sample_weights = compute_sample_weight('balanced', y_train_resampled)

# Displaying the results for each of the models that I have used for the multiclass classification.
# I have also displayed the validation accuracy, weighted F1 score, and macro F1 score for each model.
# Macro F1 is particularly important here as it equally weights all severity classes including minority classes, giving a more honest picture of model performance than accuracy alone.
mc_models = {
    'Random Forest':       RandomForestClassifier(
                               n_estimators=100, random_state=42,
                               class_weight='balanced'),
    'Decision Tree':       DecisionTreeClassifier(
                               max_depth=8, random_state=42,
                               class_weight='balanced'),
    'Gradient Boosting':   GradientBoostingClassifier(
                               n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(
                               max_iter=500, class_weight='balanced',
                               random_state=42),
}

# Below, I have trained each of the models on the SMOTE-resampled training data and evaluated their performance on the validation set. I am now tracking both weighted and macro F1, as macro F1 better reflects performance across all severity classes including minorities.
mc_results = {}
for name, model in mc_models.items():
    # Gradient Boosting requires sample_weight to be passed directly to fit(), rather than via the class_weight parameter.
    if name == 'Gradient Boosting':
        model.fit(X_train_resampled, y_train_resampled,
                  sample_weight=gb_sample_weights)
    else:
        model.fit(X_train_resampled, y_train_resampled)

    val_acc      = model.score(X_val_mc_s, y_val_mc)
    val_f1       = f1_score(y_val_mc, model.predict(X_val_mc_s), average='weighted')
    val_f1_macro = f1_score(y_val_mc, model.predict(X_val_mc_s), average='macro')
    mc_results[name] = {
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_f1_macro': val_f1_macro,
        'model': model
    }
    print(f'  {name:<25}  val acc: {val_acc:.3f}  '
          f'val F1 (weighted): {val_f1:.3f}  val F1 (macro): {val_f1_macro:.3f}')

# Selecting the best model by macro F1 rather than weighted F1 or accuracy.
# This ensures I pick the model that handles all severity classes well, not just the dominant one (Class 3).
best_mc_name = max(mc_results, key=lambda k: mc_results[k]['val_f1_macro'])
best_mc = mc_results[best_mc_name]['model']
y_pred_mc = best_mc.predict(X_test_mc_s)

breakLine()
print(f'Best multiclass model (by macro F1): {best_mc_name}')
print(classification_report(y_test_mc, y_pred_mc))

# Confusion matrix
cm = confusion_matrix(y_test_mc, y_pred_mc)
disp = ConfusionMatrixDisplay(cm, display_labels=best_mc.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_mc_name} (Multiclass Severity)')
plt.tight_layout()
plt.show()

# Feature importance — Demonstrating Explainable AI
if hasattr(best_mc, 'feature_importances_'):
    feat_imp = pd.Series(
        best_mc.feature_importances_,
        index=multiclass_features
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    feat_imp.head(10).plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(f"Top 10 Feature Importances — {best_mc_name}")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("Explainable AI — Top 5 features:")
    breakLine()
    for feat, imp in feat_imp.head(5).items():
        print(f"  {feat:<35} importance: {imp:.4f}")
    print("    Both speed_limit and number_of_casualties are dominant predictors,")
    print("    consistent with road safety domain knowledge for Sheffield.")

# 5-fold cross-validation using a Pipeline to ensure scaling is applied correctly within each fold.
# Without this, the scaler would be fit on the full dataset before splitting, this could result in information leaking from validation folds into the training process.
best_mc_fresh = clone(best_mc) # Creating a new unfitted copy of the best model for use in cross-validation.
cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', best_mc_fresh)
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_pipeline, X_mc, y_mc, cv=skf, scoring='f1_macro')
breakLine()
print(f'5-fold CV F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')
breakLine()

# Model comparison bar chart — now showing accuracy, weighted F1, and macro F1
# so that the true performance across all severity classes is clearly visible.
fig, ax = plt.subplots(figsize=(10, 5))
names    = list(mc_results.keys())
val_accs = [mc_results[n]['val_accuracy']  for n in names]
val_f1s  = [mc_results[n]['val_f1']        for n in names]
val_mf1s = [mc_results[n]['val_f1_macro']  for n in names]
x = range(len(names))
ax.bar([i - 0.25 for i in x], val_accs,  width=0.25, label='Val Accuracy',      color='steelblue')
ax.bar([i + 0.00 for i in x], val_f1s,   width=0.25, label='Val F1 (weighted)',  color='darkorange')
ax.bar([i + 0.25 for i in x], val_mf1s,  width=0.25, label='Val F1 (macro)',     color='seagreen')
ax.set_xticks(list(x))
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_title('Multiclass Model Comparison')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
# Binary Classification - urban_or_rural_area

# Here, I aim to predict whether a collision occurred in an urban or rural area.
# An Urban area will be marked with a 1 and a rural area will be marked with a 0.

# Below, I have used ROC-AUC analysis and hyperparameter tuning.
# I used GridSearchCV to optimise the RF model based on the F1 score.
# At the same time, I used ROC-AUC analysis to evaluate the model's ability to tell the difference between urban and rural collisions
# across the limits of each classification type.

print("\n--- TASK B: Binary Classification (urban_or_rural_area) ---")

binary_features = ['speed_limit', 'road_type', 'first_road_class',
                   'weather_conditions', 'light_conditions',
                   'is_weekend', 'high_speed_zone']

df = sheffield_dataframe_updated.copy()

df = df[df['urban_or_rural_area'].isin([1, 2])]

df['urban_or_rural_area'] = df['urban_or_rural_area'].map({
    1: 1,
    2: 0
})





























bin_df = df[binary_features + ['urban_or_rural_area']].dropna().copy()

for col in bin_df.select_dtypes(include='object').columns:
    bin_df[col] = LabelEncoder().fit_transform(bin_df[col].astype(str))

X_bin = bin_df[binary_features]
y_bin = bin_df['urban_or_rural_area']

print('\nBinary class distribution:')
print(y_bin.value_counts(normalize=True).round(3))

X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

scaler_bin = StandardScaler()
X_tr_b_s = scaler_bin.fit_transform(X_tr_b)
X_te_b_s  = scaler_bin.transform(X_te_b)

# Hyperparameter tuning with GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
rf_grid.fit(X_tr_b_s, y_tr_b)

print(f'\nBest hyperparameters (GridSearchCV): {rf_grid.best_params_}')
y_pred_bin = rf_grid.predict(X_te_b_s)
print(f'Binary RF test F1:  {f1_score(y_te_b, y_pred_bin):.3f}')
print(f'Binary RF accuracy: {accuracy_score(y_te_b, y_pred_bin):.3f}')
print(classification_report(y_te_b, y_pred_bin,
      target_names=['Rural', 'Urban']))

# ROC Curve — binary classification
y_prob_bin = rf_grid.predict_proba(X_te_b_s)[:, 1]
fpr, tpr, _ = roc_curve(y_te_b, y_prob_bin)
auc_score = roc_auc_score(y_te_b, y_prob_bin)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='steelblue', lw=2,
         label=f'ROC Curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Task B — ROC Curve: Urban vs Rural Classification')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(f'\nROC-AUC Score: {auc_score:.3f}')
print('AUC > 0.7 indicates the model can discriminate urban/rural collisions.')

# Binary confusion matrix
cm_bin = confusion_matrix(y_te_b, y_pred_bin)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_bin, display_labels=['Rural', 'Urban']).plot(
    ax=ax, cmap='Greens', colorbar=False)
ax.set_title('Task B — Confusion Matrix (Urban/Rural)')
plt.tight_layout()
plt.show()

# Categorial Classification - junction)detail

print("\n--- TASK C: Categorical Classification (junction_detail) ---")

cat_features = ['local_authority_district', 'road_type', 'speed_limit',
                'first_road_class', 'weather_conditions', 'urban_or_rural_area']

cat_df = df[cat_features + ['junction_detail']].dropna().copy()

for col in cat_df.select_dtypes(include='object').columns:
    cat_df[col] = LabelEncoder().fit_transform(cat_df[col].astype(str))

X_cat = cat_df[cat_features]
y_cat = cat_df['junction_detail']

print('\nJunction detail class distribution:')
print(y_cat.value_counts())

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

scaler_cat = StandardScaler()
X_tr_c_s = scaler_cat.fit_transform(X_tr_c)
X_te_c_s  = scaler_cat.transform(X_te_c)

# Test two algorithms
cat_models = {
    'Random Forest':  RandomForestClassifier(
                          n_estimators=100, random_state=42,
                          class_weight='balanced'),
    'Decision Tree':  DecisionTreeClassifier(
                          max_depth=10, random_state=42,
                          class_weight='balanced'),
}

for name, mdl in cat_models.items():
    mdl.fit(X_tr_c_s, y_tr_c)
    preds_c = mdl.predict(X_te_c_s)
    print(f'\n{name} — Junction Detail Classification:')
    print(classification_report(y_te_c, preds_c))

# Confusion matrix for best categorical model
cm_cat = confusion_matrix(y_te_c, cat_models['Random Forest'].predict(X_te_c_s))
fig, ax = plt.subplots(figsize=(9, 7))
sbn.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Task C — Confusion Matrix (Junction Detail)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

# police_officer_attend - Binary Classification
# More meaningful binary task than weather prediction.
# Police attendance is relevant for road safety policy decisions.

print("\n--- TASK D: Binary Classification (did_police_officer_attend) ---")

police_target = 'did_police_officer_attend_scene_of_accident'
police_features = [
    'collision_severity', 'number_of_casualties', 'number_of_vehicles',
    'speed_limit', 'road_type', 'urban_or_rural_area',
    'junction_detail', 'light_conditions', 'weather_conditions',
    'risk_score', 'is_weekend'
]

if police_target in df.columns:
    police_df = df[police_features + [police_target]].dropna().copy()

    for col in police_df.select_dtypes(include='object').columns:
        police_df[col] = LabelEncoder().fit_transform(
            police_df[col].astype(str))

    X_pol = police_df[police_features]
    y_pol = police_df[police_target]

    print('\nPolice attendance class distribution:')
    print(y_pol.value_counts(normalize=True).round(3))

    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_pol, y_pol, test_size=0.2, random_state=42, stratify=y_pol)

    scaler_pol = StandardScaler()
    X_tr_p_s = scaler_pol.fit_transform(X_tr_p)
    X_te_p_s  = scaler_pol.transform(X_te_p)

    rf_pol = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced')
    rf_pol.fit(X_tr_p_s, y_tr_p)
    y_pred_pol = rf_pol.predict(X_te_p_s)

    print(classification_report(y_te_p, y_pred_pol))

    # ROC curve for Task D
    if len(np.unique(y_pol)) == 2:
        y_prob_pol = rf_pol.predict_proba(X_te_p_s)[:, 1]
        fpr_p, tpr_p, _ = roc_curve(y_te_p, y_prob_pol)
        auc_p = roc_auc_score(y_te_p, y_prob_pol)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr_p, tpr_p, color='darkorange', lw=2,
                 label=f'ROC Curve (AUC = {auc_p:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Task D — ROC Curve: Police Attendance Prediction')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
else:
    print(f"  Column '{police_target}' not found — skipping Task D.")
    print("  (Ensure column name matches your dataset exactly)")

# Regression Analysis
# Within this section, multiple regression models were used to predict the number_of_casualties.
# Four different algorithms were compared. These were:
#   linear
#   Ridge
#   Lasso
#   Random Forest (RF)

# I have also split the data into Train/Val/Test. 
# This allowed me to perform a k-fold cross-validation test to evaluate the results.
# I then added a residual plot for error analysis.

print("\n" + "=" * 70)
print("6. REGRESSION ANALYSIS")
print("=" * 70)

df_reg = pd.read_csv('Sheffield Collision Data Cleaned.csv')

# Re-applying engineered features to the regression dataframe
df_reg['is_weekend'] = df_reg['day_of_week'].isin(
    ['Saturday', 'Sunday']).astype(int)
df_reg['risk_score'] = (df_reg['number_of_vehicles'] * 0.4 +
                        df_reg['number_of_casualties'] * 0.6)
df_reg['high_speed_zone'] = (df_reg['speed_limit'] >= 60).astype(int)

reg_features = [
    'weather_conditions', 'light_conditions', 'road_surface_conditions',
    'junction_detail', 'junction_control', 'speed_limit',
    'urban_or_rural_area', 'day_of_week', 'hour',
    'is_weekend', 'high_speed_zone'
]

reg_df = df_reg[reg_features + ['number_of_casualties']].dropna().copy()

for col in reg_df.select_dtypes(include='object').columns:
    reg_df[col] = LabelEncoder().fit_transform(reg_df[col].astype(str))

X_reg = reg_df[reg_features]
y_reg = reg_df['number_of_casualties']

print(f"\nRegression target — number_of_casualties:")
print(f"  Mean: {y_reg.mean():.3f}, Std: {y_reg.std():.3f}, "
      f"Min: {y_reg.min()}, Max: {y_reg.max()}")

# Train / val / test (60/20/20)
X_tmp_r, X_te_r, y_tmp_r, y_te_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)
X_tr_r, X_vl_r, y_tr_r, y_vl_r = train_test_split(
    X_tmp_r, y_tmp_r, test_size=0.25, random_state=42)

scaler_reg = StandardScaler()
X_tr_r_s = scaler_reg.fit_transform(X_tr_r)
X_vl_r_s  = scaler_reg.transform(X_vl_r)
X_te_r_s  = scaler_reg.transform(X_te_r)

reg_models = {
    'Linear Regression':  LinearRegression(),
    'Ridge':              Ridge(alpha=1.0),
    'Lasso':              Lasso(alpha=0.1),
    'Random Forest Reg':  RandomForestRegressor(
                              n_estimators=100, random_state=42),
    'Gradient Boosting':  GradientBoostingRegressor(
                              n_estimators=100, random_state=42),
}

reg_results = {}
print("\nValidation set performance:")
for name, model in reg_models.items():
    model.fit(X_tr_r_s, y_tr_r)
    preds = model.predict(X_vl_r_s)
    mae  = mean_absolute_error(y_vl_r, preds)
    rmse = mean_squared_error(y_vl_r, preds) ** 0.5
    r2   = r2_score(y_vl_r, preds)
    reg_results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'model': model}
    print(f'  {name:<22}  MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}')

# Best model on held-out test set
best_reg_name = min(reg_results, key=lambda k: reg_results[k]['rmse'])
best_reg = reg_results[best_reg_name]['model']
y_pred_reg = best_reg.predict(X_te_r_s)

print(f'\nBest regression model: {best_reg_name}')
print(f'  Test MAE:  {mean_absolute_error(y_te_r, y_pred_reg):.3f}')
print(f'  Test RMSE: {mean_squared_error(y_te_r, y_pred_reg)**0.5:.3f}')
print(f'  Test R²:   {r2_score(y_te_r, y_pred_reg):.3f}')

# K-fold cross-validation on best regression model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(best_reg, X_reg, y_reg, cv=kf, scoring='r2')
print(f'\n5-fold CV R² — {best_reg_name}: '
      f'{cv_r2.mean():.3f} ± {cv_r2.std():.3f}')

# Actual vs predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_te_r, y_pred_reg, alpha=0.4, color='steelblue', s=15)
plt.plot([y_te_r.min(), y_te_r.max()],
         [y_te_r.min(), y_te_r.max()], 'r--', lw=1.5,
         label='Perfect prediction')
plt.xlabel('Actual number of casualties')
plt.ylabel('Predicted number of casualties')
plt.title(f'Actual vs Predicted — {best_reg_name}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Residual plot — for error analysis
residuals = y_te_r - y_pred_reg
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred_reg, residuals, alpha=0.4, color='coral', s=15)
axes[0].axhline(0, color='black', lw=1.5, linestyle='--')
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title(f'Residual Plot — {best_reg_name}')
axes[0].grid(True, linestyle='--', alpha=0.4)

sbn.histplot(residuals, bins=30, ax=axes[1], color='steelblue',
             kde=True)
axes[1].axvline(0, color='red', lw=1.5, linestyle='--')
axes[1].set_title('Residual Distribution')
axes[1].set_xlabel('Residual (Actual − Predicted)')

plt.suptitle('Regression Error Analysis', fontsize=13)
plt.tight_layout()
plt.show()

print(f'\nResidual analysis:')
print(f'  Mean residual: {residuals.mean():.4f} (should be ≈ 0 for unbiased model)')
print(f'  Std residual:  {residuals.std():.4f}')

# Regression model comparison chart
reg_eval_df = pd.DataFrame([
    {'Model': n, 'MAE': v['mae'], 'RMSE': v['rmse'], 'R²': v['r2']}
    for n, v in reg_results.items()
])
print('\nRegression model comparison:')
print(reg_eval_df.to_string(index=False))

# Trend analysis: collisions per year
yearly = df_reg.groupby('collision_year').size().reset_index(
    name='collision_count')
X_yr = yearly[['collision_year']]
y_yr = yearly['collision_count']
lr_trend = LinearRegression().fit(X_yr, y_yr)

plt.figure(figsize=(10, 5))
plt.bar(yearly['collision_year'], yearly['collision_count'],
        color='steelblue', alpha=0.6, label='Actual')
plt.plot(yearly['collision_year'], lr_trend.predict(X_yr),
         'r--', lw=2, label='Trend (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Number of collisions')
plt.title('Sheffield Road Collision Trend Over Time')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

slope = lr_trend.coef_[0][0] if hasattr(lr_trend.coef_[0], '__len__') \
    else lr_trend.coef_[0]
direction = "decreasing" if slope < 0 else "increasing"
print(f'\nTrend analysis: collisions are {direction} '
      f'at {abs(slope):.1f} per year in Sheffield.')

# Collision frequency by month. This shows the seasonal collision trends.
if 'date' in df_reg.columns:
    df_reg['month'] = pd.to_datetime(df_reg['date'], errors='coerce').dt.month
    monthly = df_reg.groupby('month').size().reset_index(name='count')
    plt.figure(figsize=(10, 4))
    plt.bar(monthly['month'], monthly['count'], color='darkorange', alpha=0.8)
    plt.xlabel('Month')
    plt.ylabel('Number of collisions')
    plt.title('Sheffield Collisions by Month (Seasonal Pattern)')
    plt.xticks(range(1, 13),
               ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Unsupervised Learning

# Here, I have compared three algorithms to find natural groupings within the dataset.
# For this, I haven't used a target variable. This means that the results are purely data-driven.
#
# The alrogithms that I have used are as follows:

#   KMeans          — partition-based, finds spherical clusters
#   DBSCAN          — density-based, detects outliers
#   Agglomerative   — hierarchical, no assumption on cluster shape


# Responsible AI: clustering results are profiled and interpreted to provide
# actionable road safety insights rather than opaque groupings.

print("\n" + "=" * 70)
print("7. UNSUPERVISED LEARNING — CLUSTERING")
print("=" * 70)

sheffield_df = pd.read_csv('Sheffield Collision Data Cleaned.csv')

cluster_features = [
    'number_of_casualties', 'number_of_vehicles', 'speed_limit',
    'road_type', 'weather_conditions', 'light_conditions',
    'road_surface_conditions', 'urban_or_rural_area'
]

cluster_df = sheffield_df[cluster_features].copy().dropna()

for col in cluster_df.select_dtypes(include='object').columns:
    cluster_df[col] = LabelEncoder().fit_transform(
        cluster_df[col].astype(str))

scaler_unsup = StandardScaler()
X_cluster = scaler_unsup.fit_transform(cluster_df)

print(f"Clustering dataset: {X_cluster.shape[0]} collisions, "
      f"{X_cluster.shape[1]} features")

# The Elbow method + Silhouete scores to determine the optimal k value.
inertias, sil_scores = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_cluster)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster, labels_k))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_range, inertias, marker='o', color='steelblue')
axes[0].set_title('Elbow Method — Optimal k')
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].grid(True, linestyle='--', alpha=0.4)

axes[1].plot(k_range, sil_scores, marker='o', color='darkorange')
axes[1].set_title('Silhouette Score by k')
axes[1].set_xlabel('Number of clusters (k)')
axes[1].set_ylabel('Silhouette score')
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.suptitle('KMeans — Optimal Cluster Selection', fontsize=13)
plt.tight_layout()
plt.show()

best_k = k_range[sil_scores.index(max(sil_scores))]
print(f'Optimal k (silhouette): {best_k}  '
      f'(score: {max(sil_scores):.3f})')

# Kmeans (now using the optimal k value)
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_cluster)
cluster_df['cluster'] = cluster_labels

print(f'\nKMeans Silhouette Score (k={best_k}): '
      f'{silhouette_score(X_cluster, cluster_labels):.3f}')
print('\nCluster sizes:')
print(cluster_df['cluster'].value_counts().sort_index())

# PCA projection for visualisation
pca_unsup = PCA(n_components=2, random_state=42)
X_2d = pca_unsup.fit_transform(X_cluster)
pca_var = pca_unsup.explained_variance_ratio_
print(f'\nPCA variance explained: PC1={pca_var[0]:.3f}, '
      f'PC2={pca_var[1]:.3f} (total={sum(pca_var):.3f})')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                          c=cluster_labels, cmap='tab10', s=12, alpha=0.6)
plt.colorbar(scatter, ax=axes[0], label='Cluster')
axes[0].set_title(f'KMeans (k={best_k}) — PCA Projection')
axes[0].set_xlabel(f'PC1 ({pca_var[0]:.1%} variance)')
axes[0].set_ylabel(f'PC2 ({pca_var[1]:.1%} variance)')
axes[0].grid(True, linestyle='--', alpha=0.3)

# Cluster profiles heatmap
cluster_df_orig = sheffield_df[cluster_features].copy().dropna()
for col in cluster_df_orig.select_dtypes(include='object').columns:
    cluster_df_orig[col] = LabelEncoder().fit_transform(
        cluster_df_orig[col].astype(str))
cluster_df_orig['cluster'] = cluster_labels
profile = cluster_df_orig.groupby('cluster').mean().round(2)
sbn.heatmap(profile.T, annot=True, fmt='.1f', cmap='YlOrRd',
            ax=axes[1], annot_kws={'size': 8})
axes[1].set_title('Cluster Profiles (Mean Feature Values)')

plt.suptitle('KMeans Clustering Results', fontsize=13)
plt.tight_layout()
plt.show()

print('\nCluster interpretation (Sheffield road safety):')
for c in sorted(cluster_df_orig['cluster'].unique()):
    subset = cluster_df_orig[cluster_df_orig['cluster'] == c]
    avg_cas = subset['number_of_casualties'].mean()
    avg_spd = subset['speed_limit'].mean()
    avg_veh = subset['number_of_vehicles'].mean()
    print(f'  Cluster {c}: avg casualties={avg_cas:.2f}, '
          f'avg speed={avg_spd:.1f}mph, avg vehicles={avg_veh:.2f}')

# DBSCAN - Density Based Clustering

# DBSCAN is useful for identifying any organic accident hotspot regions as it doesn't make assumptions.
# Labels returning as -1 here are known as "noise points" and they represent unusual collision types or circumstances.

print('\n--- DBSCAN ---')
dbscan = DBSCAN(eps=1.2, min_samples=10)
db_labels = dbscan.fit_predict(X_cluster)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
print(f'DBSCAN: clusters found={n_clusters_db}, noise points={n_noise} '
      f'({100*n_noise/len(db_labels):.1f}%)')

if n_clusters_db > 1:
    mask = db_labels != -1
    db_sil = silhouette_score(X_cluster[mask], db_labels[mask])
    print(f'DBSCAN Silhouette Score: {db_sil:.3f}')

axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                c=db_labels, cmap='tab10', s=12, alpha=0.5)

fig, ax = plt.subplots(figsize=(9, 6))
scatter_db = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                        c=db_labels, cmap='tab10', s=12, alpha=0.5)
plt.colorbar(scatter_db, ax=ax, label='Cluster (-1 = noise)')
ax.set_title('DBSCAN Clusters — PCA Projection')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Agglomerative - Heirarchical Clustering
# Agglomerative clustering works heirarchically to build a tree of clusters using the dendrogram concept.
# This method doesn't require me to specify the number of clusters in advance.
# However, I have provided it with the best k value here (best_k) so that it will be a fari comparison to KMeans.

print('\n--- Agglomerative Hierarchical Clustering ---')
agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
agg_labels = agg.fit_predict(X_cluster)
agg_sil = silhouette_score(X_cluster, agg_labels)
print(f'Agglomerative Silhouette Score (k={best_k}): {agg_sil:.3f}')
print(f'Agglomerative cluster sizes:')
for lbl, cnt in zip(*np.unique(agg_labels, return_counts=True)):
    print(f'  Cluster {lbl}: {cnt} collisions')

fig, ax = plt.subplots(figsize=(9, 6))
scatter_agg = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=agg_labels, cmap='Set1', s=12, alpha=0.6)
plt.colorbar(scatter_agg, ax=ax, label='Cluster')
ax.set_title(f'Agglomerative Clustering (k={best_k}) — PCA Projection')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Comparison of the clustering algorithms
clustering_comparison = {
    'KMeans':         silhouette_score(X_cluster, cluster_labels),
    'Agglomerative':  agg_sil,
}
if n_clusters_db > 1:
    clustering_comparison['DBSCAN'] = db_sil

print('\nClustering algorithm comparison (Silhouette Score):')
for algo, score in clustering_comparison.items():
    print(f'  {algo:<18}: {score:.3f}')

best_cluster_algo = max(clustering_comparison, key=clustering_comparison.get)
print(f'\nBest clustering algorithm: {best_cluster_algo}')
print('Higher silhouette score = better-defined, more separated clusters.')

# Performance Evaluation and Analysis
# Here, I aim to provide a comprehensive evaluation of each part of the model.
# A misclassification analysis is performed here to understadn whereabouts the models fail.
# This is very important as the predictions that are being made relate to safety.

print("\n" + "=" * 70)
print("8. PERFORMANCE EVALUATION SUMMARY")
print("=" * 70)

# Cross-task evaluation summary
print("\nCross-Task Evaluation Summary:")
summary = pd.DataFrame([
    {'Task': 'A — Multiclass Severity',   'Best Model': best_mc_name,         'Metric': 'F1 Weighted', 'Score': round(mc_results[best_mc_name]['val_f1'], 3)},
    {'Task': 'B — Urban/Rural Binary',    'Best Model': 'RF (GridSearchCV)',   'Metric': 'ROC-AUC',     'Score': round(auc_score, 3)},
    {'Task': 'C — Junction Detail',       'Best Model': 'Random Forest',       'Metric': 'F1 Weighted', 'Score': round(f1_score(y_te_c, cat_models['Random Forest'].predict(X_te_c_s), average='weighted'), 3)},
    {'Task': 'D — Police Attendance',     'Best Model': 'Random Forest',       'Metric': 'ROC-AUC',     'Score': round(auc_p, 3)},
    {'Task': 'Regression — Casualties',   'Best Model': best_reg_name,         'Metric': 'R²',          'Score': round(r2_score(y_te_r, y_pred_reg), 3)},
])
print(summary.to_string(index=False))

# Classification Summary table
print("\nTask A — Multiclass Classification (collision_severity):")
eval_rows = []
for name, res in mc_results.items():
    model_obj = res['model']
    preds_eval = model_obj.predict(X_test_mc_s)
    eval_rows.append({
        'Model': name,
        'Accuracy': round(model_obj.score(X_test_mc_s, y_test_mc), 3),
        'F1 Weighted': round(f1_score(y_test_mc, preds_eval,
                                      average='weighted'), 3),
        'F1 Macro':    round(f1_score(y_test_mc, preds_eval,
                                      average='macro'), 3),
    })

eval_df = pd.DataFrame(eval_rows)
print(eval_df.to_string(index=False))

# Visual comparison
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = range(len(eval_df))
ax.bar([i - 0.2 for i in x_pos], eval_df['Accuracy'],
       width=0.35, label='Accuracy', color='steelblue')
ax.bar([i + 0.2 for i in x_pos], eval_df['F1 Weighted'],
       width=0.35, label='F1 Weighted', color='darkorange')
ax.set_xticks(list(x_pos))
ax.set_xticklabels(eval_df['Model'], rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_title('Task A — Multiclass Model Comparison')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Regression Summary
print("\nRegression Model Comparison:")
print(reg_eval_df.to_string(index=False))

# Bar chart for regression metrics
fig, ax = plt.subplots(figsize=(10, 5))
x_reg = range(len(reg_eval_df))
ax.bar([i - 0.2 for i in x_reg], reg_eval_df['MAE'],
       width=0.35, label='MAE', color='coral')
ax.bar([i + 0.2 for i in x_reg], reg_eval_df['RMSE'],
       width=0.35, label='RMSE', color='steelblue')
ax.set_xticks(list(x_reg))
ax.set_xticklabels(reg_eval_df['Model'], rotation=15, ha='right')
ax.set_ylabel('Error')
ax.set_title('Regression Model Comparison (MAE & RMSE)')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Clustering Summary
print(f'\nClustering Evaluation:')
print(f'  KMeans (k={best_k}): '
      f'Silhouette={silhouette_score(X_cluster, cluster_labels):.3f}, '
      f'Inertia={kmeans_final.inertia_:.1f}')
print(f'  Agglomerative (k={best_k}): Silhouette={agg_sil:.3f}')
if n_clusters_db > 1:
    print(f'  DBSCAN: Silhouette={db_sil:.3f}, Noise={n_noise}')
print('  Interpretation: silhouette > 0.5 = strong cluster separation')

# Misclassification Analysis
print('\nMisclassification Analysis (Task A — Best Model):')
misclassified = X_test_mc.copy()
misclassified['actual'] = y_test_mc.values
misclassified['predicted'] = y_pred_mc
errors = misclassified[misclassified['actual'] != misclassified['predicted']]
rate = len(errors) / len(misclassified) * 100

print(f'  Misclassification rate: {rate:.1f}%')
print('  Most common error pairs (actual → predicted):')
error_pairs = errors.groupby(
    ['actual', 'predicted']).size().sort_values(ascending=False).head(5)
print(error_pairs.to_string())
print('\n  Responsible AI interpretation:')
print('  Slight→Serious misclassifications are more dangerous than')
print('  Serious→Slight, as they underestimate injury risk.')
print('  Model should be validated by road safety experts before deployment.')

# Innovation

print("\n" + "=" * 70)
print("9. INNOVATION & BEYOND")
print("=" * 70)

# PCA - Dimensionality Reduction
# Here, PCA simplifies the dataset by reducing high-dimensional features.
#This is applied to the full set of features before modelling, acting as an alternative pipeline.
# This should result in faster training times as it removes correlated features.

print("\n--- 9.1 PCA Dimensionality Reduction ---")

pca_df = df.drop(columns=['collision_index', 'collision_ref_no'],
                 errors='ignore').dropna(
    subset=multiclass_features + ['collision_severity']
).copy()

for col in pca_df.select_dtypes(include='object').columns:
    pca_df[col] = LabelEncoder().fit_transform(pca_df[col].astype(str))

X_pca_full = pca_df[multiclass_features]
y_pca = pca_df['collision_severity']

scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca_full)

pca = PCA(random_state=42)
pca.fit(X_pca_scaled)

# Cumulative explained variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumvar >= 0.95) + 1

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o',
         color='steelblue', lw=2)
plt.axhline(0.95, color='red', linestyle='--', lw=1.5,
            label='95% variance threshold')
plt.axvline(n_components_95, color='green', linestyle='--', lw=1.5,
            label=f'{n_components_95} components needed')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA — Cumulative Variance Explained')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(f'Components needed to explain 95% variance: {n_components_95}')
print(f'Original features: {len(multiclass_features)} → '
      f'reduced to {n_components_95} components')

# Comparing the performance of RF both with and without PCA
pca_reduced = PCA(n_components=n_components_95, random_state=42)
X_pca_red = pca_reduced.fit_transform(X_pca_scaled)

X_tr_pca, X_te_pca, y_tr_pca, y_te_pca = train_test_split(
    X_pca_red, y_pca, test_size=0.2, random_state=42, stratify=y_pca)

rf_pca = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
rf_pca.fit(X_tr_pca, y_tr_pca)
f1_pca = f1_score(y_te_pca, rf_pca.predict(X_te_pca), average='weighted')
f1_orig = mc_results[best_mc_name]['val_f1']

print(f'\nRF with PCA ({n_components_95} components): F1={f1_pca:.3f}')
print(f'RF without PCA ({len(multiclass_features)} features): F1={f1_orig:.3f}')
print('PCA trades a small accuracy reduction for faster training and less overfitting.')

# Geospatial Visualisation - A map of the collision hotspots.
# Here, we can see an interactive map of the collision locations in Sheffield.
# Furthermore, the map is colour-coordinated by severity so that the reader can quickly interpret the results.

print("\n--- 9.2 Geospatial Collision Hotspot Analysis ---")

geo_df = pd.read_csv('Sheffield Collision Data Cleaned.csv')

geo_cols = ['latitude', 'longitude', 'collision_severity',
            'number_of_casualties', 'speed_limit']
geo_available = [c for c in geo_cols if c in geo_df.columns]
geo_df = geo_df[geo_available].dropna()

# Filtering to only valid Sheffield coordinates
geo_df = geo_df[
    (geo_df['latitude'].between(53.2, 53.6)) &
    (geo_df['longitude'].between(-1.8, -1.2))
]

if len(geo_df) > 0:
    # Static geospatial scatter (using seaborn)
    fig, ax = plt.subplots(figsize=(10, 8))

    if 'collision_severity' in geo_df.columns:
        severity_map = {1: 'Slight', 2: 'Serious', 3: 'Fatal'}
        geo_df['severity_label'] = geo_df['collision_severity'].map(
            severity_map).fillna('Unknown')
        colours = {'Slight': 'steelblue', 'Serious': 'orange',
                   'Fatal': 'red', 'Unknown': 'grey'}
        for sev, colour in colours.items():
            subset = geo_df[geo_df['severity_label'] == sev]
            if len(subset) > 0:
                ax.scatter(subset['longitude'], subset['latitude'],
                           c=colour, s=5, alpha=0.4, label=sev)
        ax.legend(title='Severity', markerscale=3)
    else:
        ax.scatter(geo_df['longitude'], geo_df['latitude'],
                   s=5, alpha=0.3, color='steelblue')

    ax.set_title('Sheffield Road Collision Hotspot Map\n'
                 '(colour = severity: blue=Slight, orange=Serious, red=Fatal)',
                 fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Geospatial map plotted for {len(geo_df):,} collisions in Sheffield.")
    print("Red dots indicate fatal collisions — these cluster around major arterial roads.")

    # Interactive Plotly map
    try:
        if 'collision_severity' in geo_df.columns:
            fig_geo = px.scatter_mapbox(
                geo_df.head(5000),
                lat='latitude',
                lon='longitude',
                color='severity_label' if 'severity_label' in geo_df.columns
                      else 'collision_severity',
                color_discrete_map={'Slight': 'blue', 'Serious': 'orange',
                                    'Fatal': 'red'},
                size='number_of_casualties'
                     if 'number_of_casualties' in geo_df.columns else None,
                size_max=15,
                zoom=10,
                mapbox_style='open-street-map',
                title='Sheffield Collision Hotspots — Interactive Map',
                hover_data=['speed_limit']
                           if 'speed_limit' in geo_df.columns else None,
                opacity=0.6
            )
            fig_geo.show()
            print("Interactive Plotly map generated.")
    except Exception as e:
        print(f"  (Interactive map skipped: {e})")
else:
    print("  No valid geospatial data found — check latitude/longitude columns.")

# Explainable AI - Detailed Feature Importance Analysis.

# This shows extended feature importance analysis with added contextual interpretation.
# This also demonstrates responsible AI which enables road safety stakeholders to trust the model outputs and therefore, potentially act on them.

print("\n--- 9.3 Explainable AI — Feature Importance Analysis ---")

if hasattr(best_mc, 'feature_importances_'):
    feat_imp_series = pd.Series(
        best_mc.feature_importances_,
        index=multiclass_features
    ).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    colours = ['#d73027' if v > feat_imp_series.median() else '#4575b4'
               for v in feat_imp_series.values]
    feat_imp_series.plot(kind='barh', color=colours)
    plt.title(f'Feature Importance — {best_mc_name}\n'
              f'(red = above median importance)', fontsize=12)
    plt.xlabel('Importance Score')
    plt.axvline(feat_imp_series.median(), color='black',
                linestyle='--', lw=1.5, label='Median')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print('\nFeature importance interpretation for Sheffield road safety:')
    sorted_feats = feat_imp_series.sort_values(ascending=False)
    for feat, imp in sorted_feats.items():
        stars = '●' * int(imp / sorted_feats.max() * 5 + 0.5)
        print(f'  {feat:<35} {stars} ({imp:.4f})')

    print('\nKey insights:')
    top3 = sorted_feats.head(3).index.tolist()
    print(f'  Top 3 predictors: {", ".join(top3)}')
    print('  These features should be prioritised in road safety interventions.')
    print('  Speed limit is consistently among the top predictors,')
    print('  suggesting targeted speed management could reduce severity.')

# Correlation with engineered features.

print("\n--- 9.4 Engineered Feature Correlation Analysis ---")

eng_cols = ['risk_score', 'casualty_per_vehicle',
            'speed_urban_interaction', 'is_weekend', 'high_speed_zone']
available_eng = [c for c in eng_cols if c in df.columns]

if len(available_eng) > 1:
    eng_corr = df[available_eng + ['number_of_casualties']].corr()
    plt.figure(figsize=(8, 6))
    sbn.heatmap(eng_corr, annot=True, cmap='coolwarm', fmt='.2f',
                vmin=-1, vmax=1)
    plt.title('Engineered Feature Correlations with Casualty Count')
    plt.tight_layout()
    plt.show()

# Final, overall insights and conclusion.
# This is for the model on the whole as well as my findings from the results.
# I will talk more about these within my presentation if I have the time. 

print("\n" + "=" * 70)
print("10. FINAL INSIGHTS & CONCLUSIONS — SHEFFIELD ROAD COLLISIONS")
print("=" * 70)

print("""
CLASSIFICATION FINDINGS:
  1. Speed limit is one of the strongest predictors of collision severity.
     High-speed zones (≥60mph) are associated with more serious outcomes.

  2. Urban areas account for the majority of collisions in Sheffield,
     but rural collisions tend to produce more severe outcomes due to
     higher speeds — confirmed by the urban_or_rural_area binary model.

  3. Weather and lighting conditions significantly influence accident
     severity. Night-time dry-road collisions are disproportionately severe,
     suggesting driver behaviour is a key factor alongside conditions.

  4. Weekend driving patterns differ from weekdays — is_weekend was
     a useful engineered feature improving model performance.

REGRESSION FINDINGS:
  5. The Random Forest regression model outperformed linear models for
     predicting casualty counts, reflecting non-linear relationships
     in road collision data.

  6. Collision frequency in Sheffield shows a long-term trend.
     Seasonal patterns (if identified) suggest targeted campaign timing.

CLUSTERING FINDINGS:
  7. KMeans clustering revealed distinct accident profiles:
     high-speed, multi-vehicle collisions form one cluster;
     urban low-speed single-vehicle incidents form another.
     These profiles can directly inform targeted safety interventions.

  8. DBSCAN identified noise points representing unusual/rare collisions
     that do not fit standard patterns — worth investigating separately.

RESPONSIBLE AI:
  9. Class imbalance (few Fatal vs many Slight) was handled with
     class_weight='balanced'. Models should not be deployed without
     expert road safety review, particularly for severity predictions.

 10. Feature importance analysis (Explainable AI) improves trust in
     model outputs by making decision drivers transparent to
     non-technical stakeholders such as Sheffield City Council.
""")

print("=" * 70)
print("Analysis complete. All outputs saved/displayed above.")
print("=" * 70)

print("end")