Tree-Based Model Visualization Toolkit
======================================

This documentation outlines the `TreeModelVis` toolkit, designed to augment the interpretability of tree-based machine learning models through advanced visualization techniques. It includes detailed insights into the `TreeModel` class, which serves as a foundation for model training and analysis, alongside functions tailored for visualizing decision paths, error rates, and sample deviations.

TreeModel Class
---------------

.. class:: TreeModel(model_type, model_params, X_train, y_train, class_names, output_dir)

   Initializes a tree-based model with specified parameters and training data.

   :param str model_type: The type of tree-based model (e.g., 'decision_tree', 'random_forest').
   :param dict model_params: Parameters for the tree-based model.
   :param DataFrame X_train: Training data features.
   :param Series y_train: Training data labels.
   :param list class_names: Names of the target classes.
   :param str output_dir: Directory to save visualizations.

   .. method:: compute_leaves_errors()

      Computes the error rates for leaves in the tree model.

   .. method:: custom_plot_tree(filename='clean_tree')

      Generates a custom visualization of the tree model.

Visualization Functions
-----------------------

.. function:: draw_path(tree_model, data_point, model_type, features)

   Visualizes the decision path for a specific data point in the model.

   :param TreeModel tree_model: The trained tree model.
   :param ndarray data_point: A single data point from the dataset.
   :param str model_type: Type of the model ('decision_tree', 'random_forest', etc.).
   :param list features: Names of the features in the data.

.. function:: leaf_to_path(tree_model, X, y, feature_names, class_names, leaf)

   Highlights the path to a specified leaf node, emphasizing decision-making criteria and errors.

.. function:: visualize_decision_tree_with_errors(tree_model, X, y, feature_names, model_type, class_names)

   Identifies and visualizes the path to the leaf node with the highest error rate, aiding in error analysis.

.. function:: visualize_decision_tree_with_nodes_large_error(tree_model, X, y, feature_names, class_names, max_error_rate, relative=False)

   Visualizes paths to nodes exceeding a specified error threshold, providing insights into model weaknesses.

.. function:: visualize_decision_tree_with_sample_deviation(tree_model, X, y, feature_names, class_names, max_error_rate)

   Focuses on nodes where the proportion of samples significantly deviates from the training distribution, highlighting potential biases or overfitting.

Each function is designed to work seamlessly with the `TreeModel` class, offering an integrated approach to model training, analysis, and visualization. This toolkit enables practitioners to gain deeper insights into their models, facilitating a more nuanced understanding of model behavior and performance.
