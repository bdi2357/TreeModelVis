import unittest
from TreeModel import TreeModel
from sklearn.model_selection import train_test_split
import pandas as pd
import os

class TestTreeModel(unittest.TestCase):
    def setUp(self):
        file_path = os.path.join("..", "data", "diabetes.csv")
        df = pd.read_csv(file_path)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.class_names = ['No', 'Yes']

    def test_random_forest_model(self):
        tree_model = TreeModel(
            model_type='random_forest',
            model_params={'max_depth': 4},
            X_train=self.X_train,
            y_train=self.y_train,
            class_names=self.class_names
        )
        dest = os.path.join("..", "graphical_output", "my_tree_visualization")
        output_path = tree_model.custom_plot_tree(filename=dest)
        # Assertions to verify the test results

if __name__ == '__main__':
    unittest.main()
