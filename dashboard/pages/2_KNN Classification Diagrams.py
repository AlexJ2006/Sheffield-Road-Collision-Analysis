import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="KNN diagrams", layout="wide")

st.title("KNN Diagrams")

# Loading the cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv("../Sheffield Collision Data Cleaned.csv")

df = load_data()

st.markdown("This page presents simple KNN visualisations using two numeric variables.")

# Selecting two numeric columns
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

col1, col2 = st.columns(2)

with col1:
    x_col = st.selectbox("X variable", numeric_cols)

with col2:
    y_col = st.selectbox("Y variable", numeric_cols, index=1)

k = st.slider("Number of neighbours (k)", 1, 15, 5)

# Dropping any NA values (not present but just a precaution)
plot_df = df[[x_col, y_col]].dropna()

# Fit KNN
knn = NearestNeighbors(n_neighbors=k)
knn.fit(plot_df)

# Use the first point as a demo for the user
query_point = plot_df.iloc[[0]]
distances, indices = knn.kneighbors(query_point)

# Creating a scater plot
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.4, label="Data")

neighbors = plot_df.iloc[indices[0]]

ax.scatter(
    neighbors[x_col],
    neighbors[y_col],
    edgecolor="black",
    s=120,
    label="Nearest neighbours"
)

ax.scatter(
    query_point[x_col],
    query_point[y_col],
    marker="X",
    s=150,
    label="Query point"
)

ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title(f"KNN diagram (k = {k})")

ax.legend()
st.pyplot(fig)
plt.close(fig)