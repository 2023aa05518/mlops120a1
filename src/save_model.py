# Model Saving Script (save_model.py)
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def save_best_model(best_params):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "best_model.pkl")
    print("Model saved as 'best_model.pkl'")


if __name__ == "__main__":
    # Example best_params for testing
    best_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 0.2
    }
    save_best_model(best_params)
