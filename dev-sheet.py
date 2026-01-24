import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sbn


#Reading my CSV (Sheffield Specific Data)
sheffield_dataframe = pd.read_csv('Collision Data - Sheffield ONLY.csv')

print("")
print(sheffield_dataframe.isnull()) #Checking which of the columns are null


# Will now individually check each column to see which ones specifically are null

for column in sheffield_dataframe.columns.isnull:
    if sheffield_dataframe[column].isnull().any():
        print(f"{column}")
print("")


#I will now check for any outliers where applicable, using a scatter graph.

print(sheffield_dataframe.shape)

print(sheffield_dataframe['collision_adjusted_severity_slight'])

sbn.scatterplot(
    data=sheffield_dataframe,    # dataframe name
    x='longitude',   # numeric variable on x-axis
    y='latitude',    # numeric variable on y-axis
    hue='longitude',    # color by class/species
    style='longitude',  # different marker style by class
    palette='Set1'      # color palette
)

plt.show()



#Columns during data pre processing that contain null values

#location_easting_osgr
#location_northing_osgr
#longitude
#latitude
#local_authority_highway_current
#collision_adjusted_severity_serious
#collision_adjusted_severity_slight

#Need to also get the data types of the items