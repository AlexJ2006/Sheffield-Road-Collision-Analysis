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

#-----------------------------------------              Local_authority_highway_current               ---------------------------------------------------

sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Updated.csv')
# INITIAL Histogram
#Drawing a histogram to view the which data is present in the column and how it is spread out.

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

#-------------------------------------------                
