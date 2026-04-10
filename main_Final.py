# main.py

from src.data_preprocessing_final import (
    load_raw_data,
    check_missing_values,
    clean_data,
    save_clean_data
)

from src.feature_engineering_final import add_features
from src.modelling_final import run_multiclass, run_regression, run_clustering
from src.evaluation_final import evaluate_classification


def main():

    # Loading and cleaning the data
    df = load_raw_data()
    check_missing_values(df)

    df = clean_data(df)
    save_clean_data(df)

    # Feature Engineering
    df = add_features(df)

    # Modelling
    model, preds, y_test = run_multiclass(df)

    run_regression(df)
    run_clustering(df)

    # Evaluation
    evaluate_classification(y_test, preds)

if __name__ == "__main__":
    main()