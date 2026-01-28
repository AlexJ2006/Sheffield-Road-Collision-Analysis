#Imports needed
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from termcolor import colored

#Reading my CSV (Sheffield Specific Data)
sheffield_dataframe = pd.read_csv('Collision Data - Sheffield ONLY.csv')

#====================== DATA PREPROCESSING ===============================================

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
print("")


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

