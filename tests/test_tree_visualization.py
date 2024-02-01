import sys
import unittest
import os

# Add the source directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'source')))

from tree_visualizer import visualize_decision_tree_with_errors
from TreeModel import TreeModel
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import webbrowser
from graphviz import Source


class TestTreeVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load and prepare dataset
        # file_path = os.path.join("..", "data", "diabetes.csv")
        if os.path.isdir(os.path.join("..", "data")):
            file_path = os.path.join("..", "data", "diabetes.csv")
        elif os.path.isdir("data"):
            file_path = os.path.join("data", "diabetes.csv")
        df = pd.read_csv(file_path)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        cls.tree_model = TreeModel(
            model_type='decision_tree',
            model_params={'max_depth': 3},
            X_train=cls.X_train,
            y_train=cls.y_train,
            class_names=['No', 'Yes']
        )

    def test_visualize_decision_tree_with_errors(self):
        dot_data = visualize_decision_tree_with_errors(
            self.tree_model, self.X_train, self.y_train, list(self.X_train.columns),
            'decision_tree', ['No', 'Yes']
        )

        # Render the DOT data to a PNG file and open it
        output_file = os.path.join("..", "graphical_output", "test_TreeModel_diabetes_error_test_set")
        graph = Source(dot_data, format="png")
        graph.render(output_file, cleanup=True)

        # Automatically open the generated image for viewing
        image_path = output_file + ".png"
        webbrowser.open('file://' + os.path.realpath(image_path))

        # Assert that the image file is created
        self.assertTrue(os.path.exists(image_path), "Tree visualization image file should exist.")


if __name__ == '__main__':
    unittest.main()
