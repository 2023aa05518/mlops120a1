import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set an experiment name (optional, but recommended)
mlflow.set_experiment("Iris Decision Tree Experiments")


def train_and_log(max_depth=None, criterion="gini"):
    with mlflow.start_run():
        # Load the Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )  # Setting a random state for reproducibility

        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("criterion", criterion)

        # Train the Decision Tree classifier
        clf = DecisionTreeClassifier(
            max_depth=max_depth, criterion=criterion, random_state=42
        )  # Setting random_state for reproducibility
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report as an artifact
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Log the trained model
        mlflow.sklearn.log_model(clf, "iris_dt_model")

        print(
            f"Run with max_depth={max_depth}, criterion={criterion} \
                completed. Accuracy: {accuracy}"
        )


if __name__ == "__main__":
    # Run experiments with different parameters
    train_and_log(max_depth=None, criterion="gini")  # Default settings
    train_and_log(max_depth=3, criterion="entropy")
    train_and_log(max_depth=5, criterion="gini")
    train_and_log(max_depth=4, criterion="entropy")

    print("All runs completed. Open MLflow UI to view results: `mlflow ui`")
