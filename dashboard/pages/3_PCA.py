import streamlit as st
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="PCA analysis", layout="wide")

st.title("Principal Component Analysis (PCA)")
st.markdown("Visualising PCA on the Sheffield collision dataset.")

# Loading the data

@st.cache_data
def load_data():
    return pd.read_csv("../Sheffield Collision Data Cleaned.csv")

df = load_data()

# Building x and y
x = df.select_dtypes(include=["number"])

x = x.drop(
    columns=[
        "collision_adjusted_severity_serious",
        "collision_adjusted_severity_slight"
    ],
    errors="ignore"
)

y = df["collision_adjusted_severity_serious"].map(
    {"Not serious": 0, "Serious": 1}
)

st.subheader("Dataset used for PCA")
st.write("Number of features used:", x.shape[1])

# Correlation Heatmap

st.subheader("Feature correlation matrix")

corr_matrix = x.corr()

fig, ax = plt.subplots(figsize=(7, 6))
sbn.heatmap(corr_matrix, cmap="coolwarm", ax=ax)
st.pyplot(fig, clear_figure=True)
plt.close(fig)

# Test, train, split.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=3
)

# Scaling

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# PCA with an interactive number of components

st.subheader("PCA settings")

n_components = st.slider(
    "Number of principal components",
    min_value=2,
    max_value=min(10, x.shape[1]),
    value=2
)

pca = PCA(n_components=n_components)
x_pca_train = pca.fit_transform(x_train_scaled)

# Explained Variance

st.subheader("Explained variance")

evr = pca.explained_variance_ratio_
cv = np.cumsum(evr)

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(range(1, len(evr) + 1), evr, alpha=0.7, label="Individual variance")
ax.plot(range(1, len(cv) + 1), cv, marker="o", label="Cumulative variance")

ax.set_xlabel("Principal components")
ax.set_ylabel("Explained variance ratio")
ax.set_title("Explained variance of PCA components")
ax.legend()

st.pyplot(fig, clear_figure=True)
plt.close(fig)

# PCA Scatter Plot

if n_components >= 2:

    st.subheader("PCA projection (PC1 vs PC2)")

    pca_df = pd.DataFrame(
        x_pca_train[:, :2],
        columns=["Principal Component 1", "Principal Component 2"]
    )

    pca_df["Serious collision"] = y_train.values

    fig, ax = plt.subplots(figsize=(8, 6))

    sbn.scatterplot(
        data=pca_df,
        x="Principal Component 1",
        y="Principal Component 2",
        hue="Serious collision",
        ax=ax,
        s=80
    )

    ax.set_title("PCA on Sheffield collision dataset")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# Heatmap

st.subheader("PCA loadings heatmap")

loadings = pd.DataFrame(
    pca.components_.T,
    index=x.columns,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)

fig, ax = plt.subplots(figsize=(8, 6))

sbn.heatmap(
    loadings,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    ax=ax
)

ax.set_title("PCA loadings")
st.pyplot(fig, clear_figure=True)
plt.close(fig)

# Table (optional whether I want to keep it)

with st.expander("Show PCA loadings table"):
    st.dataframe(loadings.round(3))