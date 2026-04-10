import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Reading the updated CSV file that contains the preprocessed, cleaned dataset.
#This was cleaned in the file named "preprocessing-graphs"
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Cleaned.csv')

print(sheffield_dataframe_updated.shape) #Printing the how many columns there are in the dataset

print(sheffield_dataframe_updated.head()) #Printing the first 5 rows for the dataset

sheffield_dataframe_updated.describe()  #Describing the dataset (how many rows and columns)

sbn.pairplot(sheffield_dataframe_updated[['speed_limit', 'number_of_vehicles']])

#Creating the feature and label variables

x = sheffield_dataframe_updated[['weather_conditions']]
y = sheffield_dataframe_updated['collision_severity']

# I have used a correlation heatmap from a previous week, rather than re-building it.

