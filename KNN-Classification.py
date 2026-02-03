import pandas as pd
import seaborn as sbn
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Updated.csv')

print(sheffield_dataframe_updated.shape) #Printing the how many columns there are in the dataset

print(sheffield_dataframe_updated.head()) #Printing the first 5 rows for the dataset

x = sheffield_dataframe_updated.drop(columns=['collision_index'], axis = 1) #Dropping the main column and keeping the rest of the columns

y = sheffield_dataframe_updated['collision_index']   #Label Column stored in Y

#Splitting the data
x_train, x_test, y_train, y_test = skl.train_test_split(x,y,random_state = 4)       

#Creating the model called KNN
KNN = train_test_split(n_neighbors = 4) #Working with 4 neighbours

#Adding the training data to the model.
KNN.fit(x_train, y_train)

#Testing how the model has learnt from the data.
KNN.score(x_test, y_test)

#Seeing how the data can predict results based on what it has learnt.
y_prediction = KNN.predict(x_test)

#Feeding some data into the program to train it.
new_dataframe = pd.DataFrame([[]]), columns = x_train.columns

#Outputting the prediction based on the data.
KNN.predict(new_dataframe)

#Improving model accuracy
scaler = MinMaxScaler() #Performing MinMax
x_train_scaled = scaler.fit_transform(x_train) #On the x_train data.
x_test_scaled = scaler.fit_transform(x_test) #And the x_test data.

KNN.fit(x_train_scaled, y_train)

KNN.score(x_test_scaled, y_test)

#Plotting the KNN decision boundary
#location_northing_osgr VS location_easting_osgr
#Latitude VS Longitude
#collision_severity VS number_of_vehicles
#collision_severity VS number_of_casualties
#collision_severity VS time
#collision_severity VS road_type
#collision_severity VS speed_limit
#collision_severity VS weather_conditions
#collision_severity VS road_surface_conditions


def plot_iris_knn(X, y, n_neighbors, weights):
    X_mat = X[['petal_length', 'petal_width']].values
    y_mat = y.values

  # Creating color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_mat,y_mat)

  # Plotting the decision boundary by assigning a color in the color map to each mesh point.

    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    #print(xx)
    #print(yy)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

  # Putting the result into a color plot
    Z = Z.reshape(xx.shape)
    #print(Z)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

  # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FF0000', label='versicolor')
    patch1 = mpatches.Patch(color='#00FF00', label='setosa')
    patch2 = mpatches.Patch(color='#0000FF', label='virginica')

    plt.legend(handles=[patch0, patch1, patch2])
    plt.xlabel('petal_length (cm)')
    plt.ylabel('petal_width (cm)')
    plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights))
    plt.show()

plot_iris_knn(x_train, y_train, 6, 'uniform')







