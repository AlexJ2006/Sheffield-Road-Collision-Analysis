import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

#   Reading the DataSet that has been cleaned
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Cleaned.csv')

#   Displaying the first 20 rows
print(sheffield_dataframe_updated.head(20))

#INSERT COLUMNS TO BE DROPPED
x = sheffield_dataframe_updated.drop([],axis=1)

#
y =sheffield_dataframe_updated['']

#Displaying a Correlation Heatmap (based on the x values entered above)

corr_matrix = x.corr()

plt.figure(figsize=(6,5))
sbn.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

#Splitting the data into Training and Test sets.


