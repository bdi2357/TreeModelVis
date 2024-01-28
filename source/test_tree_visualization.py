import os

print(os.getcwd())
from TreeModel import TreeModel  # Import your TreeModel class
from tree_visualizer import draw_path  # Import the draw_path function
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and prepare your dataset
file_path = os.path.join("..", "data", "diabetes.csv")
df = pd.read_csv(file_path)
target_column = 'Outcome'
X = df.drop(target_column, axis=1)
features = list(X.columns)
y = df[target_column]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the TreeModel instance
tree_model = TreeModel(
    model_type='random_forest',  # or another model type
    model_params={'max_depth': 4},
    X_train=X_train,
    y_train=y_train,
    class_names=['No', 'Yes']
)
# tree_model.train()  # Train the model (assuming a train method exists)

# Select a data point for visualization (e.g., the first data point from the test set)
data_point = X_test.iloc[0]

# Call draw_path with the trained model and the data point
graph = draw_path(tree_model, data_point, model_type='random_forest', features=features)

# Render and view the graph
graph.render(os.path.join('..', 'graphical_output', 'test_tree_visualization'), view=True,
             format='png')  # Specify your output path
