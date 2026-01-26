# Here, I will begin completing the data pre-processing functions.

# Need to identify which columns need pre-processing. For this, I plan on looping round each column of the databse and seeing whether they have any n/a values.

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
print(colored("Below are the columns that contain null data:", 'red'))
print("")
for column in sheffield_dataframe.columns:
    if sheffield_dataframe[column].isnull().any():
        print(f"{column}")
print("")

null_columns = sheffield_dataframe[column].isnull().any()
print("")
print(colored("Followed By the locations of the missing data:", 'red'))
print("")
print(sheffield_dataframe['location_easting_osgr'])
print("")
print(sheffield_dataframe['location_northing_osgr'])
print("")
print(sheffield_dataframe['longitude'])
print("")
print(sheffield_dataframe['latitude'])
print("")
print(sheffield_dataframe['local_authority_highway_current'])
print("")
print(sheffield_dataframe['collision_adjusted_severity_serious'])
print("")
print(sheffield_dataframe['collision_adjusted_severity_slight'])
print("")

#Reading a new CSV so that I can remove the data from it.
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Updated.csv')

#Removing Geospatial N/A data as these can't be imputed.
sheffield_dataframe_updated = sheffield_dataframe_updated.dropna(subset=[
    "latitude",
    "longitude",
    "location_easting_osgr",
    "location_northing_osgr"
])
#Above, I have had to ensure that the program understands I want to update the dataset.

#Reprinting the columsn containing N/A values to check that removal has worked.
for column in sheffield_dataframe_updated.columns:
    if sheffield_dataframe_updated[column].isnull().any():
        print(f"{column}")
print("")