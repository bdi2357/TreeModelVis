# TreeModelVis: Enhanced Tree-Based Model Visualization

TreeModelVis is a versatile Python toolkit for visualizing and customizing tree-based models, including decision trees
and ensembles like Random Forests and Gradient Boosting. Engineered for seamless integration with scikit-learn,
TreeModelVis delivers enhanced interpretability and detailed visualization capabilities, making it an indispensable tool
for model analysis and presentation.

## Features

- **Versatile Visualization**: Create detailed, customizable graphics to visualize decision trees and tree ensembles.
  ![Versatile Visualization Example](graphical_output/test_TreeModel_AsthmaDiseasePrediction.png)

- **Out-of-Sample Data Analysis**: Evaluate out-of-sample data and compare its error alignment with the training error,
  offering insights into the model's generalization.
  ![Out-of-Sample Data Analysis](graphical_output/decision_tree_with_large_error.png)

- **Data Distribution Insights**: Gain an understanding of data distribution and model decision patterns to improve
  interpretability.
  ![Data Distribution Insights](graphical_output/decision_tree_with_large_deviation.png)

- **Individual Decision Logic Analysis**: Utilize `draw_path` to visualize and understand the decision-making process
  for specific instances, providing deep insights into how models make predictions.
  ![Individual Decision Logic Example](graphical_output/test_tree_diabetes_draw_path.png)

- **Worst Performing Path Identification**: Identify and analyze the paths within the model that result in the highest
  error rates, helping to pinpoint areas for improvement.
  ![Worst Performing Path Example](graphical_output/decision_tree_worst_path.png)

- **scikit-learn Integration**: Seamlessly integrates with existing scikit-learn workflows for a smooth user experience.

- **User-Friendly**: Designed with simplicity in mind, enabling both beginners and experienced practitioners to leverage
  its capabilities efficiently.

## Quickstart

To get started with TreeModelVis, install the package using pip:

```bash
pip install TreeModelVis
```

Or clone the repository and install the requirements:

```
git clone https://github.com/yourusername/TreeModelVis.git
cd TreeModelVis
pip install -r requirements.txt
```

## Example Usage

For a hands-on introduction to TreeModelVis, including how to visualize decision trees and analyze model performance,
check out our [Example Notebook](example/Example_TreeModelVis.ipynb).

This notebook guides you through the steps of using TreeModelVis with practical examples, showcasing how to leverage the
toolkit for enhanced model visualization and interpretation. Whether you're a beginner or an experienced practitioner,
the notebook provides valuable insights into making the most out of TreeModelVis capabilities.



