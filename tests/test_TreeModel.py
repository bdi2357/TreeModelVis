import unittest
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'source')))

from TreeModel import TreeModel


class TestTreeModel(unittest.TestCase):
    def setUp(self):
        # Load the first dataset (diabetes)
        if os.path.isdir(os.path.join("..", "data")):
            file_path = os.path.join("..", "data", "diabetes.csv")
        elif os.path.isdir("data"):
            file_path = os.path.join("data", "diabetes.csv")
        df_diabetes = pd.read_csv(file_path)
        X_diabetes = df_diabetes.drop('Outcome', axis=1)
        y_diabetes = df_diabetes['Outcome']
        self.X_train_diabetes, self.X_test_diabetes, self.y_train_diabetes, self.y_test_diabetes = train_test_split(
            X_diabetes, y_diabetes, test_size=0.2, random_state=42)
        self.class_names_diabetes = ['No', 'Yes']

        # Load the second dataset (AsthmaDiseasePrediction)
        # file_path_AsthmaDiseasePrediction = os.path.join("..", "data", "AsthmaDiseasePrediction.csv")
        if os.path.isdir(os.path.join("..", "data")):
            file_path_AsthmaDiseasePrediction = os.path.join("..", "data", "AsthmaDiseasePrediction.csv")
        elif os.path.isdir("data"):
            file_path_AsthmaDiseasePrediction = os.path.join("data", "AsthmaDiseasePrediction.csv")
        df_AsthmaDiseasePrediction = pd.read_csv(file_path_AsthmaDiseasePrediction)
        X_AsthmaDiseasePrediction = df_AsthmaDiseasePrediction.drop('Difficulty-in-Breathing', axis=1)
        y_AsthmaDiseasePrediction = df_AsthmaDiseasePrediction['Difficulty-in-Breathing']
        self.X_train_AsthmaDiseasePrediction, self.X_test_AsthmaDiseasePrediction, self.y_train_AsthmaDiseasePrediction, self.y_test_AsthmaDiseasePrediction = train_test_split(
            X_AsthmaDiseasePrediction, y_AsthmaDiseasePrediction, test_size=0.2, random_state=42)
        self.class_names_AsthmaDiseasePrediction = ['0', '1']

    def test_random_forest_model_diabetes(self):
        tree_model = TreeModel(
            model_type='decision_tree',
            model_params={'max_depth': 3},
            X_train=self.X_train_diabetes,
            y_train=self.y_train_diabetes,
            class_names=self.class_names_diabetes
        )
        filename = os.path.join("..", "graphical_output", "test_TreeModel_diabetes")
        output_path = tree_model.custom_plot_tree(filename=filename)
        print(output_path)
        # Add assertions as needed

    def test_random_forest_model_AsthmaDiseasePrediction(self):
        tree_model = TreeModel(
            model_type='random_forest',
            model_params={'max_depth': 4},
            X_train=self.X_train_AsthmaDiseasePrediction,
            y_train=self.y_train_AsthmaDiseasePrediction,
            class_names=self.class_names_AsthmaDiseasePrediction
        )
        filename = os.path.join("..", "graphical_output", "test_TreeModel_AsthmaDiseasePrediction")
        output_path = tree_model.custom_plot_tree(filename=filename)
        print(output_path)
        # Add assertions as needed


if __name__ == '__main__':
    unittest.main()
