import mlflow
import mlflow.data
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import tempfile
import os


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris Decision Tree Experiments")
# Uncomment if using a remote MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")


def perform_hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Returns:
        dict: Best hyperparameters.
    """
    param_grid = {
        "max_depth": [3, 4, 5, None],
        "criterion": ["gini", "entropy"],
    }
    # Use cv=2 to handle imbalanced classes
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=2,  # Adjusted to avoid class imbalance issues
        scoring="accuracy",
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_


def train_and_log(
    max_depth=None, criterion="gini", target_column="species"
):
    with mlflow.start_run():
        data_path = "./data/iris_dataset.csv"
        try:
            columns = ['sepal length (cm)', 'sepal width (cm)',
                       'petal length (cm)', 'petal width (cm)', 'species']
            dtype_dict = {
                col: "float64" if col != "species" else "category"
                for col in columns
                }
            df = pd.read_csv(data_path, dtype=dtype_dict)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {data_path}")
            return
        print("Columns in CSV:", df.columns)  # Debugging
        mlflow.log_input(mlflow.data.from_pandas(df), context=f"{data_path}")

        # Print the column names for debugging
        # print("Columns in CSV:", df.columns)

        try:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
        except KeyError as e:
            print("Error: Column '" + str(e) + "' not found. "
                  "Available columns: " + str(df.columns.tolist()))

            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Perform hyperparameter tuning
        best_params, best_score = (
            perform_hyperparameter_tuning(X_train, y_train)
        )
        # Log best parameters and tuning details
        mlflow.log_param("best_max_depth", best_params["max_depth"])
        mlflow.log_param("best_criterion", best_params["criterion"])
        mlflow.log_metric("cv_accuracy", best_score)

        # Train model with best hyperparameters
        clf = DecisionTreeClassifier(
            max_depth=best_params["max_depth"],
            criterion=best_params["criterion"],
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        mlflow.log_metric("test_accuracy", accuracy)

        # Handle temporary file properly
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=".txt",
                                         mode="w") as tmp:
            tmp.write(report)
            tmp.flush()
            mlflow.log_artifact(tmp.name)
        # Ensure the file is closed before deletion
        os.unlink(tmp.name)

        mlflow.sklearn.log_model(clf, "iris_dt_model")

        print(f"Run completed with Test Accuracy: {accuracy}")


if __name__ == "__main__":
    train_and_log(
        max_depth=None, criterion="gini", target_column="species"
    )
    train_and_log(max_depth=3, criterion="entropy", target_column="species")
    train_and_log(max_depth=4, criterion="gini", target_column="species")
    train_and_log(max_depth=4, criterion="entropy", target_column="species")

    print("All runs completed. ")
