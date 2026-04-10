# src/feature_engineering.py
import pandas as pd

def add_features(df):
    """
    Adds engineered features to improve model performance.
    """

    # --- Time-based features ---
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )

    # --- Risk feature ---
    df['risk_score'] = (
        df['number_of_vehicles'] * 0.4 +
        df['number_of_casualties'] * 0.6
    )

    # --- Ratio feature ---
    df['casualty_per_vehicle'] = (
        df['number_of_casualties'] / (df['number_of_vehicles'] + 1)
    )

    # --- Interaction feature ---
    df['speed_urban_interaction'] = (
        df['speed_limit'] * df['urban_or_rural_area']
    )

    print("Feature engineering complete.")

    return df