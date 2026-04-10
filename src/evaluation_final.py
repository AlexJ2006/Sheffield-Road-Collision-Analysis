# src/evaluation.py

from sklearn.metrics import accuracy_score, f1_score


def evaluate_classification(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")

    