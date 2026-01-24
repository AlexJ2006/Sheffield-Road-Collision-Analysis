# Here, I will begin completing the data pre-processing functions.

# Need to identify which columns need pre-processing. For this, I plan on looping round each column of the databse and seeing whether they have any n/a values.

#Imports needed
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sbn
from termcolor import colored

#Reading my CSV (Sheffield Specific Data)
sheffield_dataframe = pd.read_csv('Collision Data - Sheffield ONLY.csv')

#====================== DATA PREPROCESSING ===============================================

print("")
print(colored("Below are the columns that contain null data:", 'red'))
print("")
for column in sheffield_dataframe.columns:
    if sheffield_dataframe[column].isnull().any():
        print(f"{column}")
print("")

null_columns = sheffield_dataframe[column].isnull().any()

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

