import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

mlflow.set_experiment("Iris Decision Tree Experiments")


def train_and_log(
    max_depth=None, criterion="gini", target_column="species"
):  # Added target_column as parameter
    with mlflow.start_run():
        data_path = "./data/iris_dataset.csv"
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {data_path}")
            return

        # Print the column names for debugging
        print("Columns in CSV:", df.columns)

        try:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
        except KeyError as e:
            print(
                f"Error: Column '{e}' not found in the CSV. \
                    Available columns are: {df.columns.tolist()}"
            )
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("target_column", target_column)

        clf = DecisionTreeClassifier(
            max_depth=max_depth, criterion=criterion, random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)

        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        mlflow.sklearn.log_model(clf, "iris_dt_model")

        print(
            f"Run with max_depth={max_depth}, criterion={criterion}, \
                target_column={target_column} completed. Accuracy: {accuracy}"
        )


if __name__ == "__main__":
    # Run experiments with different parameters
    train_and_log(
        max_depth=None, criterion="gini", target_column="species"
    )
    # Provide the correct target column name
    train_and_log(max_depth=3, criterion="entropy", target_column="species")
    train_and_log(max_depth=5, criterion="gini", target_column="species")
    train_and_log(max_depth=4, criterion="entropy", target_column="species")

    print("All runs completed. Open MLflow UI to view results: `mlflow ui`")
