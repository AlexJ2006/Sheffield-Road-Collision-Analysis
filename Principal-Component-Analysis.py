import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
import plotly.express as px

#   Reading the DataSet that has been cleaned
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Cleaned.csv')

#   Displaying the first 20 rows
print(sheffield_dataframe_updated.head(20))

#INSERT COLUMNS TO BE DROPPED   ==========================================================================
x = sheffield_dataframe_updated.drop([],axis=1)

#   ======================================================================================================
y =sheffield_dataframe_updated['']

#Displaying a Correlation Heatmap (based on the x values entered above)

corr_matrix = x.corr()

plt.figure(figsize=(6,5))
sbn.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

#Splitting the data into Training and Test sets.

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=3)


#Using the Standard Scaler Function
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#PCA

pca = PCA(n_components=2)
x_pca_train = pca.fit_transform(x_train_scaled)
x_pca_test = pca.transform(x_test_scaled)


#EXPLAINED VARIANCE
#evr = Explained Variance Ratio
#cv = Cumulative Variance
evr = pca.explained_variance_ratio_
cv = np.cumsum(pca.explained_variance_ratio_)
print(evr)
print(cv)
#Variances are shown in %'s

#   Value 1 for cv is PC1 variance and Value 2 is cumulative of PC1 and PC2

#Plot Explained Variance
plt.figure(figsize=(8,5))
plt.bar(range(1, len(evr)+1), evr,
        alpha=0.6, color='skyblue', label='Individual PC variance')
plt.plot(range(1, len(cv)+1), cv,
         marker='o', color='red', label='Cumulative variance')
plt.xticks([1,2])
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance of PCA Components (2 PCs)")
plt.legend()
plt.grid(True)
plt.show()

#Visualising the values that have been calculated within the previous step
#(Explained Varience & Cumulative Variance)

#Creating a PCA DataFrame
pca_sheffield_dataframe_updated = pd.DataFrame(x_pca_train, columns=['Principal Component 1', 'Principal Component 2'] )
pca_sheffield_dataframe_updated['categorical_column'] = y

#Plotting the PCA
import seaborn
plt.figure(figsize=(8,6))
seaborn.scatterplot(
    data=pca_sheffield_dataframe_updated,
    x='Principal Component 1',
    y='Principal Component 2',
    hue='fruit_name',                               #NEEDS CHANGING
    s=100
)
plt.title('PCA on Fruits Dataset')                  #NEEDS CHANGING
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='fruit_name')                      #NEEDS CHANGING
plt.show()

loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=['PC1', 'PC2']
)


#PCA HEATMAP
plt.figure(figsize=(8, 6))
sbn.heatmap(
    loadings,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt=".2f"
)
plt.title('PCA Loadings Heatmap')
plt.xlabel('Principal Components')
plt.ylabel('Original Features')
plt.show()

#PCA Interactive Plot
# Create DataFrame for PCA
pca_df = pd.DataFrame(X_pca_train, columns=['PC1', 'PC2'])
pca_df['fruit_name'] = y  # replace with your label column

# Define a custom color palette for better contrast
custom_colors = ['#1f77b4',  # blue
                 '#ff7f0e',  # orange
                 '#2ca02c',  # green
                 '#d62728',  # red
                 '#9467bd',  # purple
                 '#8c564b',  # brown
                 '#e377c2',  # pink
                 '#7f7f7f',  # grey
                 '#bcbd22',  # olive
                 '#17becf']  # cyan

# Interactive plot
fig = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='fruit_name',
    symbol='fruit_name',
    color_discrete_sequence=custom_colors,
    hover_data=pca_df.columns,
    title="Interactive 2D PCA of Fruits Dataset",
    width=800,
    height=600
)

fig.show()

#BUILDING KNN CLASSIFIER

knn = KNeighborsClassifier(n_neighbors=4)



#TRAINING THE KNN CLASSIFIER
knn.fit(x_pca_train, y_train)

#TESTING THE KNN CLASSIFIER
y_prediction = knn.predict(x_pca_test)

print("Accuracy", accuracy_score(y_test, y_prediction))

#Plotting the KNN
# Create a mesh grid
x_min, x_max = x_pca_train[:, 0].min() - 1, x_pca_train[:, 0].max() + 1
y_min, y_max = x_pca_train[:, 1].min() - 1, x_pca_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on grid points
Z_str = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Convert string labels to numerical values for contour
# First, get unique string labels and create a mapping
unique_labels = np.unique(y_train) # Use y_train to ensure all possible labels are included
label_to_int = {label: i for i, label in enumerate(unique_labels)}

# Apply the mapping to Z_str
Z_numeric = np.array([label_to_int[label] for label in Z_str])
Z = Z_numeric.reshape(xx.shape)

# Plot contour
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
sbn.scatterplot(x=x_pca_train[:,0], y=x_pca_train[:,1], hue=y_train, s=100)
plt.title("KNN Classification on PCA-reduced Fruits Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


#   =========================================================================   IMPROVING THE CLASSIFIER    =============================================================

pca = PCA(n_components= 0.95)             #Inputting how much variance you would like to capture

x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

#Build a KNN for this again
