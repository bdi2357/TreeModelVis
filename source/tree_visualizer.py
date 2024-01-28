from sklearn.tree import export_graphviz
import graphviz
import re
import numpy as np


def draw_path(tree_model, data_point, model_type, features):
    # Access the classifier and feature/class names from TreeModel
    UNDEF = -2
    clf = tree_model.model
    # features = tree_model.feature_names
    class_names = tree_model.class_names

    # Handle different model types
    if model_type == 'random_forest':
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
                line = line.replace("<br/>class = Yes", "")
                line = line.replace("<br/>class = No", "")
            else:
                line = line.replace("<br/>class = Yes", "class = Yes")
                line = line.replace("<br/>class = No", "class = No")
            print("after\n%s" % line)

            # Highlight the decision path
            if "->" not in line and "[label=" in line:
                node_id = line.split(" ")[0]
                if int(node_id) in np.where(decision_path == 1)[0]:
                    line = line.replace("fillcolor=\"#", "fillcolor=green, original_fillcolor=\"#")
                else:
                    line = line.replace("fillcolor=\"#", "fillcolor=white, original_fillcolor=\"#")
        new_dot_lines.append(line)

    # Reassemble the cleaned-up dot data
    clean_dot_data = "\n".join(new_dot_lines)

    # Create and return graph from dot data
    graph = graphviz.Source(clean_dot_data)
    return graph
