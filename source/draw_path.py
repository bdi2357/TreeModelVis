from sklearn.tree import export_graphviz
import graphviz
import numpy as np
import re
import os
from TreeModel import TreeModel
from sklearn.model_selection import train_test_split
import pandas as pd


def draw_path(tree_model, data_point, model_type, features):
    """
    Generates a Graphviz graph highlighting the decision path for a given data point.

    :param tree_model: Instance of TreeModel containing the trained model.
    :param data_point: Data point for which the decision path is to be visualized.
    :param model_type: Type of the model ('decision_tree', 'random_forest', etc.)
    :param features: List of feature names.
    :return: Graphviz Source object.
    """
    clf = tree_model.model
    class_names = tree_model.class_names
    tree_clf = clf.estimators_[0] if model_type != 'decision_tree' else clf

    decision_path = tree_clf.decision_path(np.array(data_point).reshape(1, -1)).toarray()[0]
    dot_data = export_graphviz(
        tree_clf,
        out_file=None,
        filled=True,
        rounded=True,
        feature_names=features,
        class_names=class_names,
        special_characters=True
    )

    clean_dot_data = process_dot_string(dot_data, decision_path, class_names)
    return graphviz.Source(clean_dot_data, format='png')


def process_dot_string(dot_string, decision_path, class_names):
    """
    Processes and modifies the DOT string for enhanced visualization.

    :param dot_string: The original DOT string.
    :param decision_path: Array indicating the decision path of the tree.
    :param class_names: Class names for the model.
    :return: Modified DOT string.
    """
    new_dot_lines = []
    for line in dot_string.split("\n"):
        if "->" not in line and "[label=" in line:
            node_id = line.split(" ")[0]
            line = modify_dot_line(line, decision_path, tree_model, node_id)
        new_dot_lines.append(line)
    return "\n".join(new_dot_lines)


def modify_dot_line(line, decision_path, tree_model, node_id):
    """
    Modifies a single line of DOT data for visual enhancements including highlighting
    decision paths and displaying error rates.

    :param line: A line from the DOT data.
    :param decision_path: Array indicating the decision path of the tree.
    :param tree_model: The TreeModel object containing error rate information.
    :param node_id: The ID of the current node.
    :return: Modified line.
    """
    line = clean_label_attributes(line, tree_model.class_names)
    line = highlight_decision_path(line, node_id, decision_path)
    line = add_error_rate_info(line, tree_model, node_id)
    return line


def add_error_rate_info(line, tree_model, node_id):
    """
    Adds error rate information to the node label.

    :param line: A line from the DOT data.
    :param tree_model: The TreeModel object containing error rate information.
    :param node_id: The ID of the current node.
    :return: Line with error rate information added.
    """
    if tree_model.model_type != 'decision_tree':
        tree_clf = tree_model.model.estimators_[0]  # Assuming use of the first tree in the forest
    else:
        tree_clf = tree_model.model  # For other model types, use the classifier directly
    if tree_clf.tree_.children_left[int(node_id)] == tree_clf.tree_.children_right[int(node_id)] == -1:
        # It's a leaf node, add error rate info
        error_rate = tree_model.leaves[int(node_id)]['errors'] / tree_model.leaves[int(node_id)]['total']
        percent_total = 100 * tree_model.leaves[int(node_id)]['total'] / tree_model.total_num_of_samples
        error_info = f"Error Rate: {error_rate:.2f}<br/> % of Total: {percent_total:.2f}%"
        for class_name in tree_model.class_names:
            line = line.replace("label=<<br/><br/>%s" % class_name, f"label=<{class_name}<br/>{error_info}<br/>")
    else:
        for class_name in tree_model.class_names:
            line = line.replace("<br/><br/>%s" % class_name, "")

    return line


def clean_label_attributes(line, class_names):
    """
    Cleans up label attributes in a DOT line.

    :param line: A line from the DOT data.
    :param class_names: Class names for the model.
    :return: Cleaned line.
    """
    line = re.sub(r'samples = \[?[0-9, ]+\]?', '', line)
    line = re.sub(r'value = \[?[0-9, ]+\]?', '', line)
    line = re.sub(r'gini = [0-9]+\.[0-9]+', '', line)
    line = re.sub(r'\[?[0-9,\. ]+\]', '', line)
    for class_name in class_names:
        line = line.replace(f"<br/>class = {class_name}", f"{class_name}")
        line = line.replace(f"<br/>{class_name}", f"{class_name}")
        line = line.replace(f"<br/>{class_name}", f"{class_name}")
    return line


def highlight_decision_path(line, node_id, decision_path):
    """
    Highlights the decision path in a DOT line.

    :param line: A line from the DOT data.
    :param node_id: The ID of the current node.
    :param decision_path: Array indicating the decision path of the tree.
    :return: Line with highlighted decision path.
    """
    if int(node_id) in np.where(decision_path == 1)[0]:
        line = line.replace("fillcolor=\"#", "fillcolor=green, original_fillcolor=\"#")
    else:
        line = line.replace("fillcolor=\"#", "fillcolor=white, original_fillcolor=\"#")
    return line


if __name__ == "__main__":
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
        model_type='random_forest',
        model_params={'max_depth': 3},
        X_train=X_train,
        y_train=y_train,
        class_names=['No', 'Yes']
    )
    # self.tree_model.train()  # Train the model (if a train method exists)

    # Select a data point for visualization
    data_point = X_test.iloc[0]
    graph = draw_path(tree_model, data_point, model_type='random_forest', features=features)
    output_file = os.path.join('..', 'graphical_output', 'test_draw_path')
    graph.render(output_file, view=True, format='png')  # Specify your output path
