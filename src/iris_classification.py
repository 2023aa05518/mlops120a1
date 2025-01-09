### Training Script (train.py)
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna

# Step 1: Hyperparameter Tuning with Optuna
def optuna_hyperparameter_tuning(data_path):
    def objective(trial):
        # Load dataset
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 20, log=True)
        min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best parameters:", study.best_params)
    print("Best accuracy:", study.best_value)

    return study.best_params

# Train and Log Model with MLflow
def train_and_log_model(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_params = optuna_hyperparameter_tuning(data_path)

    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model logged in MLflow.")

if __name__ == "__main__":
    data_path = "./data/iris_dataset.csv"  # Path to your dataset
    train_and_log_model(data_path)