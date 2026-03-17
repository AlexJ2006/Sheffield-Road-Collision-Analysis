import pandas as pd
import seaborn as sbn
import matplotlib as plt

#Reading the updated CSV file that contains the preprocessed, cleaned dataset.
#This was cleaned in the file named "preprocessing-graphs"
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Cleaned.csv')

print(sheffield_dataframe_updated.shape) #Printing the how many columns there are in the dataset

print(sheffield_dataframe_updated.head()) #Printing the first 5 rows for the dataset

sheffield_dataframe_updated.describe()  #Describing the dataset (how many rows and columns)

sbn.pairplot()

#Displaying a Correlation Heatmap (based on the x values entered above)
corr_matrix = x.corr()

plt.figure(figsize=(6,5))
sbn.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()


