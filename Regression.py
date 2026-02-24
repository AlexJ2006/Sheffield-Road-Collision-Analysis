import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

sheffield_dataframe = pd.read_csv('Sheffield Collision Data Cleaned.csv')

# SANITY CHECKING
print("")
print(sheffield_dataframe.shape)    #   Displaying the number of columns and rows within the dataset.
print(sheffield_dataframe.isnull().sum())   #   How many items within each column are null? - result shows none.
print("")


#       PAIR PLOT

#sbn.pairplot(sheffield_dataframe, plot_kws={'color': 'orange'})

#       CORRELATION HEATMAP

x = sheffield_dataframe.select_dtypes(include=["number"])

x = x.drop(columns=["collision_adjusted_severity_serious","collision_adjusted_severity_slight"],
    errors="ignore"
)   #Dropping the items present within the column that I will use as the label.

y =sheffield_dataframe['collision_adjusted_severity_serious'].map({"Not serious": 0, "Serious": 1}) #As an integer so it gives a numeric value
print(y.isna().sum())   #Ensuring none of the y values are null.

#Displaying a Correlation Heatmap (based on the x values entered above)
corr_matrix = x.corr()

plt.figure(figsize=(14, 14))
sbn.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

#       Strongly Correlated

#   collision_year & collision_injury_based
#   enhanced_severity_collision & collision_injury_based
#   attend_scene_of_accident & enhanced_severity_collision
#   collision_year & collision_injury_based

x = sheffield_dataframe[['collision_year']]
y = sheffield_dataframe[['collision_injury_based']]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=4)

linear_regression_model = LinearRegression()


