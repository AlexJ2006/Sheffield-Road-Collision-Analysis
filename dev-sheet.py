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