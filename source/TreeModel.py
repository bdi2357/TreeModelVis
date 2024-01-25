import os
import re
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from graphviz import Digraph
import pandas as pd
from sklearn.model_selection import train_test_split

class TreeModel:
    def __init__(self, model_type, model_params, X_train, y_train, class_names, output_dir='tree_visualizations'):
        """
        Initialize the TreeModel.

        :param model_type: Type of the tree-based model (e.g., 'decision_tree', 'random_forest').
        :param model_params: Dictionary of parameters for the specified model.
        :param X_train: Training data features.
        :param y_train: Training data labels.
        :param output_dir: Directory where the tree visualization will be saved.
        """
        self.model_type = model_type
        self.model_params = model_params
        self.X_train = X_train
        self.y_train = y_train
        self.output_dir = output_dir
        self.class_names =  class_names
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize the model
        self.model = self._create_model()

    def _create_model(self):
        """
        Create the model based on the model_type.
        """
        if self.model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            clf =  DecisionTreeClassifier(**self.model_params)
            clf.fit(self.X_train,self.y_train)
            return clf

        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            clf =  RandomForestClassifier(n_estimators=1, **self.model_params)
            clf.fit(self.X_train, self.y_train)
            return clf

        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(n_estimators=1, **self.model_params)

        elif self.model_type == 'extra_trees':
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=1, **self.model_params)

        elif self.model_type == 'ada_boost':
            from sklearn.ensemble import AdaBoostClassifier
            return AdaBoostClassifier(n_estimators=1, **self.model_params)

        elif self.model_type == 'hist_gradient_boosting':
            from sklearn.experimental import enable_hist_gradient_boosting  # noqa
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_iter=1, **self.model_params)

        elif self.model_type == 'bagging':
            from sklearn.ensemble import BaggingClassifier
            return BaggingClassifier(n_estimators=1, **self.model_params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def custom_plot_tree(self, filename='clean_tree'):
        """
        Plot the trained tree model with custom formatting.

        :param filename: The filename (without extension) for the saved tree visualization.
        :return: Full path of the saved visualization.
        """
        if not self.model:
            raise ValueError("Model has not been trained yet.")

        if isinstance(self.model, (DecisionTreeClassifier, RandomForestClassifier)):
            # Extract the decision tree from the model
            decision_tree = self.model if isinstance(self.model, DecisionTreeClassifier) else self.model.estimators_[0]

            # Export the decision tree to a dot file
            dot_data = export_graphviz(
                decision_tree,
                out_file=None,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names=self.X_train.columns,
                class_names=self.class_names,
                impurity=False,
                label='none'
            )

            # Clean the dot data
            clean_dot_data = self._clean_dot_data(dot_data)

            # Create a graph from the clean dot data
            graph = graphviz.Source(clean_dot_data)
            graph.format = 'png'

            # Build full file path
            file_path = os.path.join(self.output_dir, filename)
            graph.render(file_path, cleanup=True)
            return f'{file_path}.png'
        else:
            raise ValueError(f"Custom plot not supported for model type: {self.model_type}")

    def _clean_dot_data(self, dot_data):
        """
        Clean the dot data to remove unwanted information.

        :param dot_data: The raw dot data.
        :return: Cleaned dot data.
        """
        new_dot_lines = []
        clf = self.model
        if (self.model_type == "decision_tree"):
            tree = clf.tree_
        else :
            clf = self.model.estimators_[0]
            tree = clf.tree_
        for line in dot_data.split("\n"):
            # Remove unwanted lines from the label attribute


            if "->" not in line and "[label=" in line:
                node_id = line.split(" ")[0]
                # errors = leaf_errors.get(node, {'errors': 0, 'total': 0})
                UNDEF = -2
                if not (clf.tree_.children_left[int(node_id)] + clf.tree_.children_right[
                    int(node_id)] == UNDEF ):
                    print(line)
                    line = line.replace("<br/>Yes", "")
                    line = line.replace("<br/>No", "")
                    line = re.sub(r'samples = .*?\[?[0-9, ]+\]?', '', line)
                    line = re.sub(r'value = \[?[0-9, ]+\]?', '', line)
                    line = re.sub(r'gini = [0-9]+\.[0-9]+', '', line)
                    line = re.sub(r'\[?[0-9,\. ]+\]', '', line)
                    line = re.sub(r'<br/>[0-9]+<br/>', '', line)
                    line = line.replace("fillcolor=\"#", "fillcolor=white, original_fillcolor=\"#")

                    print("after")
                    print(line)
                    print("#" *44)
                else:
                    print("leaf\n%s"%line)
                    line = re.sub(r'\[?[0-9,\. ]+\]', '', line)
                    line = re.sub(r'label=<[0-9]+','label=<',line)
                    line = re.sub(r'<br/><br/>', '', line)

                    print("leaf_AFTER\n%s"%line)




            # ... (additional regex replacements as needed)


            new_dot_lines.append(line)
        return "\n".join(new_dot_lines)

if __name__ == "__main__" :
    #print(os.getcwd())

    # Example usage
    # Assuming X_train and y_train are defined
    file_path = os.path.join("..","data","diabetes.csv")
    df = pd.read_csv(file_path)

    # Define the features and the target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_names = ['No', 'Yes']
    tree_model = TreeModel(
        model_type= 'random_forest',
        #model_type='decision_tree',
        model_params={'max_depth': 4},
        X_train=X_train,
        y_train=y_train,
        class_names= class_names
    )
    dest = os.path.join("..","graphical_output","my_tree_visualization")
    output_path = tree_model.custom_plot_tree(filename=dest)
    """

    dot = Digraph()
    dot.node('A', 'Node A')
    dot.node('B', 'Node B')
    dot.edges(['AB'])
    dot.render('test-output/graph-test', view=True)
    """