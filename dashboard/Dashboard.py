import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Config
st.set_page_config(page_title="Sheffield Collision Dashboard", layout="wide")

# Sidebar
st.sidebar.title("Pre-processing")

st.sidebar.markdown(
    """
    This section contains the
    data exploration tools for the
    cleaned Sheffield collision data.
    """
)

# Loading the cleaned dataset
@st.cache_data
def load_raw_data():
    return pd.read_csv("../Sheffield Collision Data Updated.csv")

@st.cache_data
def load_clean_data():
    return pd.read_csv("../Sheffield Collision Data Cleaned.csv")

raw_df = load_raw_data()
clean_df = load_clean_data()

# Main Page
st.title("Sheffield Collision Data – Interactive Graph Builder")

st.subheader("Dataset selection")

dataset_choice = st.radio(
    "Choose which version of the data to view",
    ["Before cleansing", "After cleansing"],
    horizontal=True
)

if dataset_choice == "Before cleansing":
    df = raw_df
else:
    df = clean_df

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
all_columns = df.columns.tolist()

# Control panel
col1, col2, col3 = st.columns(3)

with col1:
    chart_type = st.selectbox(
        "Select chart type",
        [
            "Scatter plot",
            "Line plot",
            "Bar chart",
            "Count plot",
            "Box plot",
            "Histogram",
            "KDE plot"
        ]
    )

with col2:
    if chart_type in ["Scatter plot", "Line plot", "Bar chart", "Histogram", "KDE plot"]:
        x_options = numeric_columns
    else:
        x_options = all_columns

    if "x_col" in st.session_state and st.session_state["x_col"] not in x_options:
        st.session_state["x_col"] = x_options[0]

    x_col = st.selectbox(
        "Select X variable",
        x_options,
        key="x_col"
    )
# Only show the Y selector when needed
y_required = chart_type not in ["Histogram", "KDE plot", "Count plot"]

with col3:
    if y_required:
        if chart_type in ["Scatter plot", "Line plot", "Bar chart", "Box plot"]:
            y_options = numeric_columns
        else:
            y_options = all_columns

        if "y_col" in st.session_state and st.session_state["y_col"] not in y_options:
            st.session_state["y_col"] = y_options[0]

        y_col = st.selectbox(
            "Select Y variable",
            y_options,
            key="y_col"
        )
    else:
        y_col = None
        st.markdown("Y variable not required for this chart.")
# Plot Area
fig, ax = plt.subplots(figsize=(9, 6))

try:
    if chart_type == "Scatter plot":
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)

    elif chart_type == "Line plot":
        sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)

    elif chart_type == "Bar chart":
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)

    elif chart_type == "Count plot":
        sns.countplot(data=df, x=x_col, ax=ax)

    elif chart_type == "Box plot":
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)

    elif chart_type == "Histogram":
        sns.histplot(data=df, x=x_col, bins=40, ax=ax)

    elif chart_type == "KDE plot":
        sns.kdeplot(data=df[x_col].dropna(), ax=ax)

    if y_col is None:
        ax.set_title(f"{chart_type} – {x_col}")
    else:
        ax.set_title(f"{chart_type} – {x_col} vs {y_col}")

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error("This chart cannot be created with the selected variables.")
    st.exception(e)

# Previewing the data
with st.expander("Show data preview"):
    st.dataframe(df.head(50))