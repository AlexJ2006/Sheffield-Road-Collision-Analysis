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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#Reading the updated CSV file that contains the preprocessed, cleaned dataset.
#This was cleaned in the file named "preprocessing-graphs"
sheffield_dataframe_updated = pd.read_csv('Sheffield Collision Data Cleaned.csv')

print(sheffield_dataframe_updated.shape) #Printing the how many columns there are in the dataset

print(sheffield_dataframe_updated.head()) #Printing the first 5 rows for the dataset

#Cheking which column contains the time that is throwing an error.
for col in sheffield_dataframe_updated.columns:
    if sheffield_dataframe_updated[col].astype(str).str.contains(':').any():
        print(col)
#This has been determined to be the time column. 
#This now needs to be converted so that it shows just the hour rather than the hour and minute.

#Manipulating the time column to create the 'hour'.
sheffield_dataframe_updated['hour'] = pd.to_datetime(
    sheffield_dataframe_updated['time'],
    errors='coerce'
).dt.hour


#CATEGORISING THE DATA

feature_columns = ['weather_conditions', 'road_type', 'light_conditions','speed_limit',
                   'number_of_vehicles', 'road_surface_conditions', 'junction_detail',
                   'junction_control', 'urban_or_rural_area', 'day_of_week', 'hour']

x = sheffield_dataframe_updated[feature_columns]    #Using the feature columns listed above, here.

y = sheffield_dataframe_updated['collision_severity']   #Label Column stored in Y

categorical_features = ['weather_conditions', 'road_type', 'light_conditions',
                        'road_surface_conditions', 'junction_detail', 'junction_control',
                        'urban_or_rural_area', 'day_of_week']

numerical_features = ['speed_limit', 'number_of_vehicles', 'hour']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

#Splitting the data
x_train, x_test, y_train, y_test =train_test_split(x,y,random_state=4)       

from sklearn.neighbors import KNeighborsClassifier

x_train_fe = preprocessor.fit_transform(x_train)
x_test_fe  = preprocessor.transform(x_test)

KNN = KNeighborsClassifier(n_neighbors=4)

KNN.fit(x_train_fe, y_train)

#Checking the current accuracy of the model
print("KNN accuracy:", KNN.score(x_test_fe, y_test))

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


#       KNN 1                               Location (northing and easting)
def plot_collision_knn(X, y, n_neighbors, weights):

    plot_df = X.copy()
    plot_df['collision_severity'] = y.values

    plot_df = plot_df.dropna(subset=[
        'location_northing_osgr',
        'location_easting_osgr',
        'collision_severity'
    ])

    X_mat = plot_df[['location_northing_osgr',
                     'location_easting_osgr']].values

    y_mat = plot_df['collision_severity'].values

    # Creating color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)

    # Plotting the decision boundary
    mesh_step_size = 100
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(
        X_mat[:, 0],
        X_mat[:, 1],
        s=plot_symbol_size,
        c=y_mat,
        cmap=cmap_bold,
        edgecolor='black'
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('Location northing (OSGR)')
    plt.ylabel('Location easting (OSGR)')
    plt.title(
        "Collision severity KNN (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

    plt.show()

plot_collision_knn(
    sheffield_dataframe_updated[['location_northing_osgr', 'location_easting_osgr']],
    y,
    6,
    'uniform'
)


#       KNN 2                  Weather Conditions                                                 

def plot_collision_knn(X, y, n_neighbors, weights):

    plot_df = X.copy()
    plot_df['collision_severity'] = y.values

    plot_df = plot_df.dropna(subset=[
        'weather_conditions',
        'light_conditions'
    ])

    X_mat = plot_df[['weather_conditions',
                     'light_conditions']].values

    y_mat = plot_df['collision_severity'].values

    # Creating color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)

    # Plotting the decision boundary
    mesh_step_size = 1
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(
        X_mat[:, 0],
        X_mat[:, 1],
        s=plot_symbol_size,
        c=y_mat,
        cmap=cmap_bold,
        edgecolor='black'
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('Weather Conditions')
    plt.ylabel('Light Conditions')
    plt.title(
        "Collision severity KNN (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

    plt.show()

plot_collision_knn(
    sheffield_dataframe_updated[['weather_conditions', 'light_conditions']],
    y,
    6,
    'uniform'
)

#       KNN 3                               Speed Limit & Number Of Vehicles

def plot_collision_knn(X, y, n_neighbors, weights):

    plot_df = X.copy()
    plot_df['collision_severity'] = y.values

    plot_df = plot_df.dropna(subset=[
        'speed_limit',
        'number_of_vehicles'
    ])

    X_mat = plot_df[['speed_limit',
                     'number_of_vehicles']].values

    y_mat = plot_df['collision_severity'].values

    # Creating color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)

    # Plotting the decision boundary
    mesh_step_size = 1
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(
        X_mat[:, 0],
        X_mat[:, 1],
        s=plot_symbol_size,
        c=y_mat,
        cmap=cmap_bold,
        edgecolor='black'
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('speed_limit')
    plt.ylabel('number_of_vehicles')
    plt.title(
        "Collision severity KNN (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

    plt.show()

plot_collision_knn(
    sheffield_dataframe_updated[['speed_limit', 'number_of_vehicles']],
    y,
    6,
    'uniform'
)

#       KNN 4                               Road Type & Road Surface Conditions

def plot_collision_knn(X, y, n_neighbors, weights):

    plot_df = X.copy()
    plot_df['collision_severity'] = y.values

    plot_df = plot_df.dropna(subset=[
        'road_type',
        'road_surface_conditions'
    ])

    X_mat = plot_df[['road_type',
                     'road_surface_conditions']].values

    y_mat = plot_df['collision_severity'].values

    # Creating color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)

    # Plotting the decision boundary
    mesh_step_size = 1
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(
        X_mat[:, 0],
        X_mat[:, 1],
        s=plot_symbol_size,
        c=y_mat,
        cmap=cmap_bold,
        edgecolor='black'
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('road_type')
    plt.ylabel('road_surface_conditions')
    plt.title(
        "Collision severity KNN (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

    plt.show()

plot_collision_knn(
    sheffield_dataframe_updated[['road_type', 'road_surface_conditions']],
    y,
    6,
    'uniform'
)

#       KNN 4                               Day Of The Week & Area Type (Urban Or Rural)

def plot_collision_knn(X, y, n_neighbors, weights):

    plot_df = X.copy()
    plot_df['collision_severity'] = y.values

    plot_df = plot_df.dropna(subset=[
        'day_of_week',
        'urban_or_rural_area'
    ])

    X_mat = plot_df[['day_of_week',
                     'urban_or_rural_area']].values

    y_mat = plot_df['collision_severity'].values

    # Creating color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)

    # Plotting the decision boundary
    mesh_step_size = 1
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(
        X_mat[:, 0],
        X_mat[:, 1],
        s=plot_symbol_size,
        c=y_mat,
        cmap=cmap_bold,
        edgecolor='black'
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('day_of_week')
    plt.ylabel('urban_or_rural_area')
    plt.title(
        "Collision severity KNN (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

    plt.show()

plot_collision_knn(
    sheffield_dataframe_updated[['day_of_week', 'urban_or_rural_area']],
    y,
    6,
    'uniform'
)

#       KNN 5                               Junction Detail & Junction Control

def plot_collision_knn(X, y, n_neighbors, weights):

    plot_df = X.copy()
    plot_df['collision_severity'] = y.values

    plot_df = plot_df.dropna(subset=[
        'junction_detail',
        'junction_control'
    ])

    X_mat = plot_df[['junction_detail',
                     'junction_control']].values

    y_mat = plot_df['collision_severity'].values

    # Creating color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)

    # Plotting the decision boundary
    mesh_step_size = 2
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(
        X_mat[:, 0],
        X_mat[:, 1],
        s=plot_symbol_size,
        c=y_mat,
        cmap=cmap_bold,
        edgecolor='black'
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('junction_detail')
    plt.ylabel('junction_control')
    plt.title(
        "Collision severity KNN (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

    plt.show()

plot_collision_knn(
    sheffield_dataframe_updated[['junction_detail', 'junction_control']],
    y,
    6,
    'uniform'
)

#   All of my KNN's are plotted against Collision Severity.

#   Working on the KNN Elbow Method
k_values = range(1, 21)
error_rates = []

for k in k_values:
    knn_elbow = KNeighborsClassifier(n_neighbors=k)

    knn_elbow.fit(x_train_fe, y_train)

    y_pred = knn_elbow.predict(x_test_fe)

    error = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error)

plt.figure(figsize=(8, 5))
plt.plot(k_values, error_rates, marker='o')
plt.xlabel("Number of neighbours (k)")
plt.ylabel("Error rate")
plt.title("Elbow Method for KNN (Sheffield collision dataset)")
plt.grid(True)
plt.show()


