import unittest
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'source')))

from TreeModel import TreeModel
from tree_visualizer import draw_path


class TestTreeVisualizer(unittest.TestCase):

    def setUp(self):
        file_path = os.path.join("..", "data", "diabetes.csv")
        df = pd.read_csv(file_path)
        target_column = 'Outcome'
        X = df.drop(target_column, axis=1)
        self.features = list(X.columns)
        y = df[target_column]

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the TreeModel instance
        self.tree_model = TreeModel(
            model_type='random_forest',
            model_params={'max_depth': 4},
            X_train=X_train,
            y_train=y_train,
            class_names=['No', 'Yes']
        )
        # self.tree_model.train()  # Train the model (if a train method exists)

        # Select a data point for visualization
        self.data_point = X_test.iloc[0]

    def test_draw_path(self):
        # Call draw_path with the trained model and the data point
        graph = draw_path(self.tree_model, self.data_point, model_type='random_forest', features=self.features)
        output_file = os.path.join('..', 'graphical_output', 'test_tree_diabetes_draw_path')
        graph.render(output_file, view=True, format='png')  # Specify your output path

        # Assertions or additional checks can be added here
        self.assertTrue(os.path.exists(output_file + '.png'), "Graph image file was not created.")


if __name__ == '__main__':
    unittest.main()
