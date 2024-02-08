from sklearn.tree import export_graphviz
import graphviz
import re
import numpy as np
import os
from TreeModel import TreeModel
from sklearn.model_selection import train_test_split
import pandas as pd
from graphviz import Source
from IPython.display import display


# from IPython.display import display, Image
def draw_path(tree_model, data_point, model_type, features):
    """
    Description: Visualizes the decision path for a specific data point within a tree-based model. It highlights the path taken by the data point through the model's decision nodes.
    Key Features:
    Supports various tree-based models including decision trees and random forest classifiers.
    Cleans and modifies the DOT graph for clearer visualization.
    Highlights the decision path in green.
    :param tree_model:
    :param data_point:
    :param model_type:
    :param features:
    :return:
    """
    # Access the classifier and feature/class names from TreeModel
    UNDEF = -2
    clf = tree_model.model
    # features = tree_model.feature_names
    class_names = tree_model.class_names

    # Handle different model types
    if model_type != 'decision_tree':
        tree_clf = clf.estimators_[0]  # Assuming use of the first tree in the forest
    else:
        tree_clf = clf  # For other model types, use the classifier directly

    decision_path = tree_clf.decision_path(np.array(data_point).reshape(1, -1)).toarray()[0]
    # print(data_point)
    # decision_path = tree_clf.decision_path(data_point).toarray()[0]

    # Generate DOT-format graph description
    dot_data = export_graphviz(
        tree_clf,
        out_file=None,
        filled=True,
        rounded=True,
        feature_names=features,
        class_names=class_names,
        special_characters=True
    )

    new_dot_lines = []
    for line in dot_data.split("\n"):
        # Clean-up label attributes
        if "->" not in line and "[label=" in line:
            node_id = line.split(" ")[0]
            print("before\n%s" % line)
            line = re.sub(r'samples = \[?[0-9, ]+\]?', '', line)
            line = re.sub(r'value = \[?[0-9, ]+\]?', '', line)
            line = re.sub(r'gini = [0-9]+\.[0-9]+', '', line)
            line = line.replace("<br/><br/><br/>", '<br/>')
            line = line.replace("<br/><br/>", '<br/>')
            line = re.sub(r'\[?[0-9,\. ]+\]', '', line)

            # line = line.replace("label=<<br/>class =","label=<class =")
            if not (tree_clf.tree_.children_left[int(node_id)] + tree_clf.tree_.children_right[
                int(node_id)] == UNDEF):
                for class_name in class_names:
                    line = line.replace("<br/>class = %s" % class_name, "")

            else:
                lbl = f"\nError Rate: {tree_model.leaves[int(node_id)]['errors'] / tree_model.leaves[int(node_id)]['total']:.2f}"
                lbl2 = f"\n % of the total:{100 * tree_model.leaves[int(node_id)]['total'] / tree_model.total_num_of_samples:.2f}%"
                # line = line.replace("label=<No>", "label=<No" + "<br/>" + lbl + "<br/>" + lbl2 + ">")
                # line = line.replace("label=<Yes>", "label=<Yes" + "<br/>" + lbl + "<br/>" + lbl2 + ">")
                for class_name in class_names:
                    line = line.replace("<br/>class = %s" % class_name,
                                        "%s" % class_name + "<br/>" + lbl + "<br/>" + lbl2)
                    line = line.replace("<br/>%s" % class_name, "%s" % class_name)
                # line = line.replace("<br/>class = No", "class = No"+"<br/>" + lbl + "<br/>" + lbl2)
            print("after\n%s" % line)

            # Highlight the decision path
            if "->" not in line and "[label=" in line:
                node_id = line.split(" ")[0]
                if int(node_id) in np.where(decision_path == 1)[0]:
                    line = line.replace("fillcolor=\"#", "fillcolor=green, original_fillcolor=\"#")
                else:
                    line = line.replace("fillcolor=\"#", "fillcolor=white, original_fillcolor=\"#")
        if 'label=' in line:
            # Increase font size
            print("HEREX")
            line = re.sub(r'fontsize=\d+', 'fontsize=20', line)
        if '[shape=box]' in line or '->' not in line:
            # Increase node size
            line = line.replace('[shape=box]', '[shape=box, width=1.8, height=0.9]')
        new_dot_lines.append(line)
        if 'graph [' in line:
            # Insert settings right after the graph declaration
            new_dot_lines.append('  node [fontsize=30, width=3, height=1.5];')

    # Reassemble the cleaned-up dot data
    clean_dot_data = "\n".join(new_dot_lines)

    # Create and return graph from dot data
    # graph = graphviz.Source(clean_dot_data)
    # graph = graphviz.Source(clean_dot_data, format='png')
    # graph.attr(dpi='300')  # Higher DPI for better resolution
    # graph.attr('node', fontsize='12')  # Increase node font size
    # graph.attr('edge', fontsize='10')  # Increase edge font

    # Create a graph from the modified dot data

    graph = graphviz.Source(clean_dot_data, format='png')

    return graph


def leaf_to_path(tree_model, X, y, feature_names, class_names, leaf):
    """
    Description: Visualizes the decision path for a specific data point within a tree-based model.
    It highlights the path taken by the data point through the model's decision nodes.
    Key Features:
    Supports various tree-based models including decision trees and random forest classifiers.
    Cleans and modifies the DOT graph for clearer visualization.
    Highlights the decision path in green.
    :param tree_model:
    :param X:
    :param y:
    :param feature_names:
    :param class_names:
    :param leaf:
    :return:
    """
    # Generate DOT-format graph description
    # leaf_errors = compute_leaf_errors(clf, X, y)
    UNDEF = -2
    TREE_LEAF = -1
    clf = tree_model.model
    if tree_model.model_type != 'decision_tree':
        tree_clf = clf.estimators_[0]  # Assuming use of the first tree in the forest
    else:
        tree_clf = clf  # For other model types, use the classifier directly
    dot_data = export_graphviz(tree_clf, out_file=None, filled=True, rounded=True,
                               feature_names=feature_names, class_names=class_names,
                               special_characters=True)

    # Initialize a list to hold the new lines of the DOT string
    decision_path = tree_clf.decision_path(X).toarray()
    new_dot_lines = []
    m = np.where(decision_path[:, leaf])[0][0]
    decision_path = decision_path[m, :]

    # Iterate through each line of the original DOT string
    for line in dot_data.split("\n"):
        # Remove 'samples' and 'value' lines from the label attribute


        line = re.sub(r'samples = \[?[0-9, ]+\]?', '', line)
        line = re.sub(r'value = \[?[0-9, ]+\]?', '', line)
        line = re.sub(r'gini = [0-9]+\.[0-9]+', '', line)
        line = re.sub(r'\[?[0-9,\. ]+\]', '', line)
        # If the line defines a node, set its fillcolor
        if "->" not in line and "[label=" in line:
            node_id = line.split(" ")[0]
            print("before\n%s" % line)
            line = re.sub(r'samples = \[?[0-9, ]+\]?', '', line)
            line = re.sub(r'value = \[?[0-9, ]+\]?', '', line)
            line = re.sub(r'gini = [0-9]+\.[0-9]+', '', line)
            line = line.replace("<br/><br/><br/>", '<br/>')
            line = line.replace("<br/><br/>", '<br/>')
            line = re.sub(r'\[?[0-9,\. ]+\]', '', line)
            # errors = leaf_errors.get(node, {'errors': 0, 'total': 0})
            if not (tree_clf.tree_.children_left[int(node_id)] + tree_clf.tree_.children_right[int(node_id)] == UNDEF):
                for class_name in class_names:
                    line = line.replace("<br/>class = %s" % class_name, "")
                print("after\n%s" % line)
            else:
                print("leaf\n%s" % line)
                lbl = f"\nError Rate: {tree_model.leaves[int(node_id)]['errors'] / tree_model.leaves[int(node_id)]['total']:.2f}"
                # lbl2 = f"\n % of the total:{100 * tree_model.leaves[int(node_id)]['total'] / tree_model.total_num_of_samples:.2f}%"
                # line = line.replace("label=<No>", "label=<No" + "<br/>" + lbl + "<br/>" + lbl2 + ">")
                # line = line.replace("label=<Yes>", "label=<Yes" + "<br/>" + lbl + "<br/>" + lbl2 + ">")
                for class_name in class_names:
                    line = line.replace("<br/>class = %s" % class_name, "%s" % class_name + "<br/>" + lbl)
            if int(node_id) in np.where(decision_path == 1)[0]:
                line = line.replace("fillcolor=\"#", "fillcolor=red, original_fillcolor=\"#")

                if node_id == str(leaf):
                    label = f"Error Rate: {tree_model.leaves[int(node_id)]['errors'] / tree_model.leaves[int(node_id)]['total']:.2f}"
                    line = line.replace("label=<<br/><br/>", "label=<<br/>%s<br/>" % label)
                    # label += f"\nWorst Leaf: {leaf_errors[int(node_id)]['errors'] / leaf_errors[int(node_id)]['total']:.2f}"


            else:
                if int(node_id) == TREE_LEAF or (
                        tree_clf.tree_.children_left[int(node_id)] + tree_clf.tree_.children_right[
                    int(node_id)] == UNDEF):
                    label = f"Error Rate: {tree_model.leaves[int(node_id)]['errors'] / tree_model.leaves[int(node_id)]['total']:.2f}"

                    line = line.replace("label=<<br/><br/>", "label=<<br/>%s<br/>" % label)
                line = line.replace("fillcolor=\"#", "fillcolor=white, original_fillcolor=\"#")

        new_dot_lines.append(line)
        if 'graph [' in line:
            # Insert settings right after the graph declaration
            new_dot_lines.append('  node [fontsize=30, width=3, height=1.5];')

    # Join the modified lines back into a single string
    new_dot_data = "\n".join(new_dot_lines)

    return new_dot_data


def visualize_decision_tree_with_errors(tree_model, X, y, feature_names, model_type, class_names):
    """
    Description: Identifies and visualizes the path leading to the leaf node with the highest error rate.
    This function is intended to help diagnose areas where the model may be underperforming.
    Key Features:
    Automatically identifies the worst-performing leaf node based on error rate.
    Utilizes leaf_to_path to visualize the path to this node.
    :param tree_model:
    :param X:
    :param y:
    :param feature_names:
    :param model_type:
    :param class_names:
    :return:
    """
    clf = tree_model.model
    # features = tree_model.feature_names
    class_names = tree_model.class_names

    # Handle different model types
    if model_type != 'decision_tree':
        tree_clf = clf.estimators_[0]  # Assuming use of the first tree in the forest
    else:
        tree_clf = clf  # For other model types, use the classifier directly

    worst_leaf = max(tree_model.leaves,
                     key=lambda x: (tree_model.leaves[int(x)]['errors'] / tree_model.leaves[int(x)]['total']) if
                     tree_model.leaves[int(x)]['total'] > 0 else 0)

    print("tree_model ", worst_leaf)
    return leaf_to_path(tree_model, X, y, feature_names, class_names, worst_leaf)


def visualize_decision_tree_with_nodes_large_error(tree_model, X, y, feature_names, class_names, max_error_rate,
                                                   relative=False):
    """
    Description: Extends error visualization by identifying multiple nodes with error rates exceeding a specified threshold.
    It generates visualizations for these nodes to analyze high-error regions within the model.
    Key Features:
    Can work on a relative scale, comparing test errors against training errors to find significant deviations.
    Generates separate path visualizations for each node exceeding the error threshold.
    :param tree_model:
    :param X:
    :param y:
    :param feature_names:
    :param class_names:
    :param max_error_rate:
    :param relative:
    :return:
    """
    # Determine if model is a single decision tree or an ensemble of trees
    UNDEF = -2
    clf = tree_model.model
    if tree_model.model_type != 'decision_tree':
        tree_clf = clf.estimators_[0]  # Use the first tree in the ensemble
    else:
        tree_clf = clf  # Use the classifier directly if it's a single decision tree

    leaf_errors = tree_model.compute_leaves_errors_extd(X, y)
    print("comapre")
    print(leaf_errors.keys())
    print(tree_model.leaves.keys())

    def tmp_f(leaf_errors, leaf):
        if not relative:
            return (leaf_errors[leaf]['errors'] / leaf_errors[leaf]['total']) if leaf_errors[leaf]['total'] > 0 else 0
        else:
            x1 = (leaf_errors[leaf]['errors'] / leaf_errors[leaf]['total']) if leaf_errors[leaf]['total'] > 0 else 0
            return x1 - tree_model.leaves[int(leaf)]['errors'] / tree_model.leaves[int(leaf)]['total']

    # Find leaves with error rate above max_error_rate
    above_error_rate = [leaf for leaf in leaf_errors if tmp_f(leaf_errors, leaf) > max_error_rate]

    # Generate DOT-format graph description
    dot_data = export_graphviz(tree_clf, out_file=None, filled=True, rounded=True,
                               feature_names=feature_names, class_names=class_names,
                               special_characters=True)

    # Initialize a list to hold the new lines of the DOT string
    new_dot_lines = []

    # Iterate through each line of the original DOT string
    for line in dot_data.split("\n"):
        if "->" not in line and "[label=" in line:
            line = re.sub(r'samples = \[?[0-9, ]+\]?', '', line)
            line = re.sub(r'value = \[?[0-9, ]+\]?', '', line)
            line = re.sub(r'gini = [0-9]+\.[0-9]+', '', line)
            line = line.replace("<br/><br/><br/>", '<br/>')
            line = line.replace("<br/><br/>", '<br/>')
            line = re.sub(r'\[?[0-9,\. ]+\]', '', line)
            node_id = line.split(" ")[0]
            if not (tree_clf.tree_.children_left[int(node_id)] + tree_clf.tree_.children_right[int(node_id)] == UNDEF):
                for class_name in class_names:
                    line = line.replace("<br/>class = %s" % class_name, "")
            else:
                print("leaf\n%s" % line)
                lbl = f"\nTrain Error Rate: {tree_model.leaves[int(node_id)]['errors'] / tree_model.leaves[int(node_id)]['total']:.2f}"
                print(node_id)
                print("#" * 50)
                print(leaf_errors.keys())
                print(tree_model.leaves.keys())
                print(node_id in leaf_errors.keys())
                print(int(node_id) in leaf_errors.keys())
                print(leaf_errors[int(node_id)]['errors'])
                print(leaf_errors[int(node_id)]['total'])
                if leaf_errors[int(node_id)]['total'] > 0:
                    lbl2 = f"\nTest Error Rate: {leaf_errors[int(node_id)]['errors'] / leaf_errors[int(node_id)]['total']:.2f}"
                else:
                    lbl2 = f"\nTest Error Rate: {0:.2f}"

                for class_name in class_names:
                    line = line.replace("<br/>class = %s" % class_name,
                                        "%s" % class_name + "<br/>" + lbl + "<br/>" + lbl2)
                    line = line.replace("<br/>%s" % class_name, "%s" % class_name)
            if int(node_id) in above_error_rate:
                error_rate = tmp_f(leaf_errors, int(node_id))
                # label = f"Error Rate: {error_rate:.2f}"
                line = line.replace("fillcolor=\"#", "fillcolor=red, original_fillcolor=\"#")
                for class_name in class_names:
                    line = line.replace("<br/>%s" % class_name, "%s" % class_name)
                print("line_err\n%s" % line)
                # line = line.replace("label=<<br/><br/>", f"label=<<br/>{label}<br/>")
            else:

                if tree_clf.tree_.feature[int(node_id)] == UNDEF:
                    error_rate = tmp_f(leaf_errors, int(node_id))
                    label = f"Error Rate: {error_rate:.2f}"
                    line = line.replace("label=<<br/><br/>", "label=<<br/>%s<br/>" % label)
                line = line.replace("fillcolor=\"#", "fillcolor=white, original_fillcolor=\"#")

        new_dot_lines.append(line)

    # Join the modified lines back into a single string
    new_dot_data = "\n".join(new_dot_lines)

    # Generate paths to individual leaves with high error rates
    paths_to_high_error_leaves = [leaf_to_path(tree_model, X, y, feature_names, class_names, leaf) for leaf in
                                  above_error_rate]

    return new_dot_data, paths_to_high_error_leaves


def visualize_decision_tree_with_sample_deviation(tree_model, X, y, feature_names, class_names, max_error_rate):
    """
    Description: Focuses on identifying and visualizing nodes where the proportion of samples significantly deviates from the training set, potentially indicating areas of the model sensitive to sample distribution changes.
    Key Features:
    Identifies nodes with significant deviations in sample proportions between training and test (or other) datasets.
    Visualizes these nodes to highlight potential overfitting or underfitting.
    :param tree_model:
    :param X:
    :param y:
    :param feature_names:
    :param class_names:
    :param max_error_rate:
    :return:
    """
    # Determine if model is a single decision tree or an ensemble of trees
    UNDEF = -2
    total_oos_samples = X.shape[0]
    clf = tree_model.model
    if tree_model.model_type != 'decision_tree':
        tree_clf = clf.estimators_[0]  # Use the first tree in the ensemble
    else:
        tree_clf = clf  # Use the classifier directly if it's a single decision tree

    leaf_errors = tree_model.compute_leaves_errors_extd(X, y)

    def tmp_f(leaf_errors, leaf):
        x1 = leaf_errors[leaf]['total'] / float(total_oos_samples)
        y1 = tree_model.leaves[int(leaf)]['total'] / tree_model.total_num_of_samples
        return abs(x1 - y1) / y1

    # Find leaves with error rate above max_error_rate
    above_error_rate = [leaf for leaf in leaf_errors if tmp_f(leaf_errors, leaf) > max_error_rate]

    # Generate DOT-format graph description
    dot_data = export_graphviz(tree_clf, out_file=None, filled=True, rounded=True,
                               feature_names=feature_names, class_names=class_names,
                               special_characters=True)

    # Initialize a list to hold the new lines of the DOT string
    new_dot_lines = []

    # Iterate through each line of the original DOT string
    for line in dot_data.split("\n"):
        if "->" not in line and "[label=" in line:
            line = re.sub(r'samples = \[?[0-9, ]+\]?', '', line)
            line = re.sub(r'value = \[?[0-9, ]+\]?', '', line)
            line = re.sub(r'gini = [0-9]+\.[0-9]+', '', line)
            line = line.replace("<br/><br/><br/>", '<br/>')
            line = line.replace("<br/><br/>", '<br/>')
            line = re.sub(r'\[?[0-9,\. ]+\]', '', line)
            node_id = line.split(" ")[0]
            if not (tree_clf.tree_.children_left[int(node_id)] + tree_clf.tree_.children_right[int(node_id)] == UNDEF):
                for class_name in class_names:
                    line = line.replace("<br/>class = %s" % class_name, "")
            else:
                print("leaf\n%s" % line)
                lbl = f"Train % of the total:{100 * tree_model.leaves[int(node_id)]['total'] / tree_model.total_num_of_samples:.2f}%"
                lbl2 = f"Test % of the total:{100 * leaf_errors[int(node_id)]['total'] / total_oos_samples :.2f}%"


                for class_name in class_names:
                    line = line.replace("<br/>class = %s" % class_name,
                                        "%s" % class_name + "<br/>" + lbl + "<br/>" + lbl2)
                    line = line.replace("<br/>%s" % class_name, "%s" % class_name)
            if int(node_id) in above_error_rate:
                error_rate = tmp_f(leaf_errors, int(node_id))
                # label = f"Error Rate: {error_rate:.2f}"
                line = line.replace("fillcolor=\"#", "fillcolor=purple, original_fillcolor=\"#")
                for class_name in class_names:
                    line = line.replace("<br/>%s" % class_name, "%s" % class_name)
                print("line_err\n%s" % line)
                # line = line.replace("label=<<br/><br/>", f"label=<<br/>{label}<br/>")
            else:

                if tree_clf.tree_.feature[int(node_id)] == UNDEF:
                    error_rate = tmp_f(leaf_errors, int(node_id))
                    label = f"Error Rate: {error_rate:.2f}"
                    line = line.replace("label=<<br/><br/>", "label=<<br/>%s<br/>" % label)
                line = line.replace("fillcolor=\"#", "fillcolor=white, original_fillcolor=\"#")

        new_dot_lines.append(line)

    # Join the modified lines back into a single string
    new_dot_data = "\n".join(new_dot_lines)

    # Generate paths to individual leaves with high error rates
    paths_to_high_error_leaves = [leaf_to_path(tree_model, X, y, feature_names, class_names, leaf) for leaf in
                                  above_error_rate if leaf_errors[leaf]['total'] > 0]

    return new_dot_data, paths_to_high_error_leaves


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
    model_type = 'decision_tree'
    class_names = ['No', 'Yes']
    tree_model = TreeModel(
        # model_type='random_forest',
        model_type='random_forest',
        model_params={'max_depth': 3},
        X_train=X_train,
        y_train=y_train,
        class_names=class_names
    )

    visualize_decision_tree_with_errors(tree_model, X, y, features, model_type, class_names)
    dot = visualize_decision_tree_with_errors(tree_model, X, y, features, model_type, class_names)
    # dot.render('decision_tree_with_errorsN2', view=True, format='png')
    # display(graph)
    graph = Source(dot)

    # Render the graph to a file (e.g., PNG)
    graph.render(filename=f"decision_tree_worst_path", format="png", cleanup=True)

    data_point = X_test.iloc[0]
    graph = draw_path(tree_model, data_point, model_type='random_forest', features=features)
    output_file = os.path.join('..', 'graphical_output', 'test_draw_path1')
    graph.render(output_file, view=True, format='png')  # Specify your output path

    dot, paths = visualize_decision_tree_with_nodes_large_error(tree_model, X_test, y_test,
                                                                feature_names=X_test.columns.tolist(),
                                                                class_names=['No', 'Yes'], max_error_rate=0.4)
    # dot.render('decision_tree_with_errorsN2', view=True, format='png')
    # display(graph)
    graph = Source(dot)

    # Render the graph to a file (e.g., PNG)
    graph.render(filename=f"decision_tree_with_large_error", format="png", cleanup=True)

    # Display the graph in Jupyter Notebook
    display(graph)

    dot, paths = visualize_decision_tree_with_nodes_large_error(tree_model, X_test, y_test,
                                                                feature_names=X_test.columns.tolist(),
                                                                class_names=['No', 'Yes'], max_error_rate=0.2,
                                                                relative=True)
    # dot.render('decision_tree_with_errorsN2', view=True, format='png')
    # display(graph)
    graph = Source(dot)

    # Render the graph to a file (e.g., PNG)
    graph.render(filename=f"decision_tree_with_large_relative_error", format="png", cleanup=True)

    # Display the graph in Jupyter Notebook
    display(graph)

    # visualize_decision_tree_with_sample_deviation
    dot, paths = visualize_decision_tree_with_sample_deviation(tree_model, X_test, y_test,
                                                               feature_names=X_test.columns.tolist(),
                                                               class_names=['No', 'Yes'], max_error_rate=0.5)
    # dot.render('decision_tree_with_errorsN2', view=True, format='png')
    # display(graph)
    graph = Source(dot)

    # Render the graph to a file (e.g., PNG)
    graph.render(filename=f"decision_tree_with_large_deviation", format="png", cleanup=True)

    # Display the graph in Jupyter Notebook
    display(graph)
