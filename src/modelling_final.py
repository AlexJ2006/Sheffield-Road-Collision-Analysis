# src/modelling.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================
# MULTICLASS MODEL
# =========================
def run_multiclass(df):

    print("\nRunning Multiclass Classification...")

    features = [
        'weather_conditions', 'road_surface_conditions', 'light_conditions',
        'speed_limit', 'number_of_vehicles', 'number_of_casualties',
        'urban_or_rural_area', 'day_of_week', 'junction_detail', 'road_type'
    ]

    df = df.dropna()

    # Encode categorical
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[features]
    y = df['collision_severity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    return model, preds, y_test


# =========================
# REGRESSION
# =========================
def run_regression(df):

    print("\nRunning Regression...")

    features = ['speed_limit', 'urban_or_rural_area', 'hour']
    target = 'number_of_casualties'

    df = df.dropna()

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Regression complete.")

    return model


# =========================
# CLUSTERING
# =========================
def run_clustering(df):

    print("\nRunning Clustering...")

    features = ['number_of_casualties', 'number_of_vehicles', 'speed_limit']

    df = df[features].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)

    print(f"Silhouette Score: {score:.3f}")

    return kmeans