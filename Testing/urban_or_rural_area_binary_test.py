import pandas as pd

df = pd.read_csv('Sheffield Collision Data Updated.csv') # Loading in the updated dataset.

print(df['urban_or_rural_area'].head(50))   # Printing the first 50 items to see whether they are binary or not
