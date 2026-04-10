# src/data_preprocessing.py

import pandas as pd

# ANSI color helper (fixed: using numeric codes)
colored = lambda text, color: f"\033[{color}m{text}\033[0m"


# =========================
# LOAD DATA
# =========================
def load_raw_data(file_path='data/raw/Collision Data - Sheffield ONLY.csv'):
    """
    Loads the raw dataset and prints basic structure info.
    """
    df = pd.read_csv(file_path)

    print("Dataset Shape:")
    print(df.shape)

    print("\nPreview:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    return df


# =========================
# DATA QUALITY CHECK
# =========================
def check_missing_values(df):
    """
    Displays columns that contain missing values.
    """

    print("\n==================================================================")
    print(colored("Columns with missing values:", "31"))  # 31 = red
    print("")

    for column in df.columns:
        if df[column].isnull().any():
            print(column)

    print("==================================================================\n")


# =========================
# OPTIONAL: FULL NULL SUMMARY
# =========================
def show_null_summary(df):
    """
    Prints full null counts for all columns.
    """
    print("\nMissing Values Summary:")
    print(df.isnull().sum())
    print("")


# =========================
# CLEAN DATA (PLACEHOLDER)
# =========================
def clean_data(df):
    """
    Apply all preprocessing steps here.
    (You will paste your fillna + cleaning logic inside this function)
    """

    # Example (replace with your real logic):
    # df['latitude'] = df['latitude'].fillna(df['latitude'].mean())

    return df


# =========================
# SAVE CLEAN DATA
# =========================
def save_clean_data(df, file_path='data/processed/Sheffield Collision Data Cleaned.csv'):
    """
    Saves cleaned dataset to processed folder.
    """
    df.to_csv(file_path, index=False)
    print(f"\nCleaned data saved to: {file_path}")