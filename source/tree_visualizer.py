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


def leaf_to_path(clf, X, y, feature_names, class_names, leaf):
    # Generate DOT-format graph description
    leaf_errors = compute_leaf_errors(clf, X, y)
    dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True,
                               feature_names=features, class_names=class_names,
                               special_characters=True)

    # Initialize a list to hold the new lines of the DOT string
    decision_path = clf.decision_path(X).toarray()
    new_dot_lines = []
    m = np.where(decision_path[:, leaf])[0][0]
    decision_path = decision_path[m, :]

    # Iterate through each line of the original DOT string
    for line in dot_data.split("\n"):
        # Remove 'samples' and 'value' lines from the label attribute
        line = re.sub(r'samples = \[?[0-9, ]+\]?', '', line)
        line = re.sub(r'value = \[?[0-9, ]+\]?', '', line)
        line = re.sub(r'gini = [0-9]+\.[0-9]+', '', line)

        # If the line defines a node, set its fillcolor
        if "->" not in line and "[label=" in line:
            node_id = line.split(" ")[0]
            # errors = leaf_errors.get(node, {'errors': 0, 'total': 0})
            if not (clf.tree_.children_left[int(node_id)] == clf.tree_.children_right[
                int(node_id)] == _tree.TREE_UNDEFINED):
                line = line.replace("<br/><br/><br/><br/>class = Yes", "")
                line = line.replace("<br/><br/><br/><br/>class = No", "")

            if int(node_id) in np.where(decision_path == 1)[0]:
                line = line.replace("fillcolor=\"#", "fillcolor=red, original_fillcolor=\"#")

                if node_id == str(leaf):
                    label = f"Error Rate: {leaf_errors[int(node_id)]['errors'] / leaf_errors[int(node_id)]['total']:.2f}"
                    # print("line")
                    # print(line)
                    # label=<<br/><br/><br/>class = Yes>
                    line = line.replace("label=<<br/><br/>", "label=<<br/>%s<br/>" % label)
                    # label += f"\nWorst Leaf: {leaf_errors[int(node_id)]['errors'] / leaf_errors[int(node_id)]['total']:.2f}"


            else:
                if int(node_id) == _tree.TREE_LEAF or (clf.tree_.children_left[int(node_id)] + clf.tree_.children_right[
                    int(node_id)] == _tree.TREE_UNDEFINED):
                    label = f"Error Rate: {leaf_errors[int(node_id)]['errors'] / leaf_errors[int(node_id)]['total']:.2f}"

                    line = line.replace("label=<<br/><br/>", "label=<<br/>%s<br/>" % label)
                line = line.replace("fillcolor=\"#", "fillcolor=white, original_fillcolor=\"#")

        new_dot_lines.append(line)

    # Join the modified lines back into a single string
    new_dot_data = "\n".join(new_dot_lines)

    return new_dot_data


def visualize_decision_tree_with_errorsN3(clf, X, y, feature_names, class_names):
    tree = clf.tree_
    leaf_classes = [np.argmax(value[0]) for value in tree.value]
    leaf_errors = compute_leaf_errors(clf, X, y)
    worst_leaf = max(leaf_errors, key=lambda x: (leaf_errors[x]['errors'] / leaf_errors[x]['total']) if leaf_errors[x][
                                                                                                            'total'] > 0 else 0)
    return leaf_to_path(clf, X, y, feature_names, class_names, worst_leaf)
