from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from divexplorer.outcomes import get_false_positive_rate_outcome
from divexplorer import DivergenceExplorer
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.tree import _tree
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import _tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.tree import export_graphviz
import graphviz
import pydot
from IPython.display import Image
import os



"""
Extracts features from a given itemset(as opposed to the specific values these features assume)
Args:
FP_fm: The dataframe produced by DivExplorer, sorted in descending order of divergence metric

Returns: Set: The features present in the maximally divergent itemset

"""

def extract_features_from_itemset(FP_fm):
  return set([condition.split('=')[0] for condition in FP_fm.iloc[0]['itemset']])



def calculate_tree_depth(tree, node, depth = 0):
    """
    Calculates the depth of a decision tree recursively.

    Args:
        tree: The DecisionTreeClassifier object representing the tree.
        node: The index of the node to start calculating depth from.
        depth: The current depth (initially 0 for the root node).

    Returns:
        The depth of the subtree rooted at the given node.
    """

    if node == -1: # Leaf node
        return depth
    left_depth = calculate_tree_depth(tree, tree.tree_.children_left[node], depth + 1)
    right_depth = calculate_tree_depth(tree, tree.tree_.children_right[node], depth + 1)
    return max(left_depth, right_depth)




"""
This function prints the decision path in the DecisionTreeClassifier corresponding to a particular itemset. It is invoked on a particular itemset, both before and after pruning to validate that pruning has occurred.
Args:
clf: The trained DecisionTreeClassifier
X: The One-Hot-Encoded representation of the particular(singular) divergent itemset under consideration
feature_names: The expanded, One-Hot-Encoded features columns names in X
"""

def debug_decision_path(clf, X, feature_names):
    decision_path = clf.decision_path(X).toarray()
    print("In debug_decision_path(), The length of X is: ", len(X))
    for sample_index in range(X.shape[0]):
        print(f"Sample {sample_index}:")
        node_indices = decision_path[sample_index].nonzero()[0]
        for node_id in node_indices:
            if clf.tree_.feature[node_id] != _tree.TREE_UNDEFINED:
                feature_index = clf.tree_.feature[node_id]
                if feature_index < len(feature_names):
                    feature_name = feature_names[feature_index]
                    threshold = clf.tree_.threshold[node_id]
                    threshold_sign = "<=" if X.iloc[sample_index][feature_name] <= threshold else ">"
                    print(f"Node {node_id}: (Feature {feature_name} {threshold_sign} {threshold})")
                else:
                    print(f"Node {node_id}: (Leaf node)")
            else:
                print("Leaf node")



def check_feature(feature_name, divergent_itemset_features):
    """
    Checks if a feature name (one-hot encoded) is present in the list of divergent itemset features(not in one-hot encoded form).

    Args:
        feature_name: The name of the feature to check (e.g., 'age_25-45').
        divergent_itemset_features: A list of features in the divergent itemset (e.g., ['#prior', 'race', 'age', 'charge']).

    Returns:
        True if the feature name starts with any of the divergent feature substrings, False otherwise.
    """

    # Convert feature name to lowercase for case-insensitive comparison
    feature_name_lower = feature_name.lower()

    # Check if the feature name starts with any of the divergent feature substrings
    for divergent_feature in divergent_itemset_features:
        if feature_name_lower.startswith(divergent_feature.lower()):
            return True

    return False


def standardize_name(feature_name):
    """
    Extracts the base feature name from a one-hot encoded feature name.

    Args:
        feature_name: The one-hot encoded feature name (e.g., 'age_25-45').

    Returns:
        The base feature name (e.g., 'age').
    """

    # Find the index of the first underscore, if it exists
    underscore_index = feature_name.find("_")

    # If an underscore is found, return the substring before it
    if underscore_index != -1:
        return feature_name[:underscore_index]

    # If no underscore is found, return the original feature name
    return feature_name




"""
This function prunes the DecisionTreeClassifier based on the most divergent itemset found by DivExplorer.
Args:
clf: The DecisionTreeClassifier
feature_names: The features present in the dataset
divergent_itemset_features: The features in the maximally divergent itemset
expanded_feature_names: One-hot-encoded column names
encoded_df: The entirety of the encoded dataframe
X_train: The training dataset

"""


def prune_tree_based_on_itemset(clf, feature_names, divergent_itemset_features, expanded_feature_names, encoded_df, X_train):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right

    assert clf.tree_.node_count > 0, "Decision tree classifier has no nodes"

    feature_decision_split = clf.tree_.feature

    # Initialize a dictionary with a hypothetical itemset for the Adult Income dataset
    divergent_itemset_values = {
        'gender': 'Male',
        'race': 'White',
        'education': 'Bachelors'  # Assuming age is a continuous variable, so we use a specific value instead of a range
    }

    # Convert the dictionary to a DataFrame
    X_debug = pd.DataFrame([divergent_itemset_values])

    # Apply pd.get_dummies() to convert categorical variables to dummy/indicator variables
    X_debug_encoded = pd.get_dummies(X_debug)

    # Reorder the columns to match the order in income_df_encoded, filling missing columns with 0
    X_debug_encoded = X_debug_encoded.reindex(columns=encoded_df.columns, fill_value=0)

    # Ensure X_debug_encoded has the same structure as the training data
    X_debug_encoded = X_debug_encoded.reindex(columns=X_train.columns, fill_value=0)

    assert not X_debug_encoded.isnull().values.any(), "DataFrame contains null values after processing"
    assert X_debug_encoded.shape[1] == X_train.shape[1], "X_debug_encoded has incorrect number of features"

    # This list will hold the indices of the nodes to prune
    nodes_to_prune = []

    feature_names = expanded_feature_names

    # Debug decision path before pruning
  #print("Decision path before pruning:")
   # debug_decision_path(clf, X_debug_encoded, expanded_feature_names)
    def recurse(node, features_in_path):
      # If it's a leaf node, no further action is necessary
      if children_left[node] == children_right[node] == -1:
          return

      # Determine the feature name of the current node
      if feature_decision_split[node] != _tree.TREE_UNDEFINED:
          feature_name = feature_names[feature_decision_split[node]]
          assert feature_name in expanded_feature_names, "Feature used for splitting not found in feature names"

      else:
          feature_name = "leaf"


      # If the current node is a split node, process further
      if feature_name != "leaf":

        current_feature = standardize_name(feature_name)
        features_in_path_copy = features_in_path.copy()  # Create a copy
        features_in_path_copy.append(current_feature)  # Add to the copy

        if divergent_itemset_features.issubset(set(features_in_path_copy)):  # Check on the copy
            nodes_to_prune.append(node)
            return
          # Add the feature of the current node to the path
        features_in_path = features_in_path + [standardize_name(feature_name)]

      # Continue recursion on the children nodes
      if children_left[node] != -1:
          recurse(children_left[node], features_in_path)
      if children_right[node] != -1:
          recurse(children_right[node], features_in_path)

      #Invoke the recurse function starting from the root node
    recurse(0, [])

    # Prune the marked nodes
    for node_to_prune in nodes_to_prune:
      clf.tree_.children_left[node_to_prune] = _tree.TREE_LEAF
      clf.tree_.children_right[node_to_prune] = _tree.TREE_LEAF
      clf.tree_.feature[node_to_prune] = _tree.TREE_UNDEFINED
      clf.tree_.threshold[node_to_prune] = -2

    print("Decision path after pruning:")
    debug_decision_path(clf, X_debug_encoded, expanded_feature_names)

    assert len(nodes_to_prune) > 0

    return clf, nodes_to_prune  # Return the modified classifier and nodes marked for pruning


def prune_tree(clf, FP_fm, expanded_feature_names, encoded_df, X_train):
    all_features = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'educational-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income']

    # Get the most divergent itemset
    most_divergent_itemset_features = extract_features_from_itemset(FP_fm)

    print("In prune_tree(), most divergent itemset features are: ", most_divergent_itemset_features)

    # Proceed with pruning based on this itemset
    pruned_clf, nodes_to_prune = prune_tree_based_on_itemset(clf, all_features, most_divergent_itemset_features, expanded_feature_names, encoded_df, X_train)

    assert clf.tree_.node_count >= pruned_clf.tree_.node_count, "Node count did not decrease after pruning"

    return pruned_clf, nodes_to_prune



def visualize_pruned_tree(dt_classifier, feature_names, class_names, pruned_nodes, output_filename='pruned_tree'):
    """Exports a GraphViz representation of a decision tree and highlights pruned nodes using pydot."""
    # Export the decision tree to a dot file
    export_graphviz(dt_classifier, out_file="tree.dot",
                    feature_names=feature_names,
                    class_names=class_names,
                    filled=True, rounded=True)

    graphs = pydot.graph_from_dot_file('tree.dot')
    if graphs is None:
        raise ValueError("Could not load dot file.")
    graph = graphs[0]

    # Iterate through nodes in the graph and highlight pruned nodes
    for node in graph.get_nodes():
        node_id = node.get_name().strip('"')
        if node_id.isdigit() and int(node_id) in pruned_nodes:
            node.set_fillcolor('#ff9999')  # Light red
            node.set_style('filled')
            node.set_shape('diamond')  # Change shape to diamond
            node.set_label("Pruned\n" + node.get_label())  # Add 'Pruned' text to label

    output_path_dot = os.path.join(os.getcwd(), f"{output_filename}.dot")
    graph.write_dot(output_path_dot)

    output_path_png = os.path.join(os.getcwd(), f"{output_filename}.png")
    graph.write_png(output_path_png)

    return Image(output_path_png)



# Load the dataset
income_df = pd.read_csv('adult.csv')  # Load adult income dataset, source: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

# Replace '?' with NaN for entire DataFrame
income_df = income_df.replace('?', pd.NA)

# Encode the 'income' column to binary format, interpretable for binary classification
income_df['income'] = income_df['income'].map({'<=50K': 0, '>50K': 1})

# One-hot encode the categorical columns
categorical_cols = ['age', 'workclass',  'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
income_df_encoded = pd.get_dummies(income_df, columns=categorical_cols)

# Define the feature matrix and the target vector
y = income_df['income']
X = income_df_encoded.drop(columns=['income'])  # Use all columns except 'income', which is the target variable

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Add the continuous columns back into X_train and X_test

continuous_cols = ['fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
X_train[continuous_cols] = income_df[continuous_cols].iloc[X_train.index]
X_test[continuous_cols] = income_df[continuous_cols].iloc[X_test.index]


expanded_feature_names = X_train.columns.tolist()
print("Expanded feature names are: ", expanded_feature_names)

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Fit the classifier to the data
dt_classifier.fit(X_train, y_train)

# Make predictions using the trained model for the entire dataset
income_df['predicted_dt'] = dt_classifier.predict(X)

# Calculate and Print the Overall False Positive Rate for income dataset
tn = (income_df['income'] == 0) & (income_df['predicted_dt'] == 0)  # True Negatives
fp = (income_df['income'] == 0) & (income_df['predicted_dt'] == 1)  # False Positives
overall_fp_rate = fp.sum() / (fp.sum() + tn.sum())
print("Overall False Positive Rate of Decision Tree Classifier Before Pruning:", overall_fp_rate)

# Create masks to filter 'income_df' for the itemset before/after pruning
itemset_mask_before = (income_df['education'] == 'Bachelors') & \
                      (income_df['race'] == 'White') & \
                      (income_df['gender'] == 'Male')

# Filter 'income_df' directly for conditional FP rate calculation
itemset_data_before = income_df[itemset_mask_before]

# Conditional FP rate on 'income_df' - the 'predicted_dt' is here
fp_before = (itemset_data_before['income'] == 0) & (itemset_data_before['predicted_dt'] == 1)
tn_before = (itemset_data_before['income'] == 0) & (itemset_data_before['predicted_dt'] == 0)
conditional_fp_rate_before = fp_before.sum() / (fp_before.sum() + tn_before.sum())

print("Conditional FP Rate on most divergent itemset (BEFORE Pruning):", conditional_fp_rate_before)

original_depth = calculate_tree_depth(dt_classifier, 0)  # Calculate depth of original tree
print("Original Decision Tree Depth:", original_depth)

y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

# Calculate ROC curve values
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print("ROC AUC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Baseline
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Unpruned Decision Tree Classifier')
plt.legend(loc="lower right")
plt.show()

# Compute the false positive rate using the provided function from DivExplorer
income_df['fp'] = get_false_positive_rate_outcome(y, income_df['predicted_dt'])

# Calculate and print the classification accuracy
train_accuracy = accuracy_score(y_train, dt_classifier.predict(X_train))
test_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Perform divergence analysis with DivExplorer for the original(unpruned) DecisionTreeClassifier
fp_diver = DivergenceExplorer(income_df)
attributes = ['education', 'occupation', 'race', 'gender']
FP_fm = fp_diver.get_pattern_divergence(min_support=0.1, attributes=attributes, boolean_outcomes=['fp'])
FP_fm = FP_fm.sort_values(by="fp_div", ascending=False, ignore_index=True)
FP_fm.head(10)

