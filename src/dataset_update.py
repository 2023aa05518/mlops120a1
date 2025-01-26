import pandas as pd

# File path of the dataset
# Replace with your DVC-tracked file path
dataset_path = "./data/iris_dataset.csv"

# Load the existing dataset
df = pd.read_csv(dataset_path)

# Print original shape
print("Original dataset shape:", df.shape)

# Define new rows to append
new_data = pd.DataFrame({
    'sepal_length': [5.9, 6.0, 6.1, 6.2, 6.3],
    'sepal_width': [3.0, 3.1, 3.2, 3.3, 3.4],
    'petal_length': [4.5, 4.6, 4.7, 4.8, 4.9],
    'petal_width': [1.5, 1.6, 1.7, 1.8, 1.9],
    'species': ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica']
})

# Append new rows
df = pd.concat([df, new_data], ignore_index=True)

# Save the updated dataset back to the same file
df.to_csv(dataset_path, index=False)

# Print updated shape
print("Updated dataset shape:", df.shape)
