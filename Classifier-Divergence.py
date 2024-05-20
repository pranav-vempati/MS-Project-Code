import sklearn
import sklearn.tree
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
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import os




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
#print("Expanded feature names are: ", expanded_feature_names)



'''
for alpha in np.linspace(0.00, 0.10, 100): 
    temp_income_df = income_df.copy()
    os.environ['ALPHA'] = str(alpha)
    model = DecisionTreeClassifier(random_state = 42, class_weight='balanced')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    f_score = f1_score(y_test, predictions)
    print(f"Alpha: {alpha}, Test F-score: {f_score}")
    tree_depth = model.get_depth()
   # print("Depth of the dynamic Decision Tree:", tree_depth)
    temp_income_df['predicted_dt'] = model.predict(X)
    temp_income_df['fp'] = get_false_positive_rate_outcome(y, temp_income_df['predicted_dt'])
    # Perform divergence analysis with DivExplorer for the original(unpruned) DecisionTreeClassifier
    fp_diver = DivergenceExplorer(temp_income_df)
    attributes = ['education', 'occupation', 'race', 'gender']
    FP_fm = fp_diver.get_pattern_divergence(min_support=0.1, attributes=attributes, boolean_outcomes=['fp'])
    FP_fm = FP_fm.sort_values(by="fp_div", ascending=False, ignore_index=True)
    print("Most 10 divergent itemsets for alpha = ", alpha)
    print(FP_fm.head(10))
'''

dt_classifier = DecisionTreeClassifier(random_state=42, class_weight='balanced')


# Fit the classifier to the data
dt_classifier.fit(X_train, y_train)

# Determine and print the depth of the tree
tree_depth = dt_classifier.get_depth()
print("Depth of the dynamic Decision Tree:", tree_depth)


# Make predictions using the trained model
y_pred_train = dt_classifier.predict(X_train)
y_pred_test = dt_classifier.predict(X_test)

# Compute the F-score
f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"Training F1 Score, alpha = 0.001: {f1_train}")
print(f"Testing F1 Score, alpha = 0.001: {f1_test}")


# Make predictions using the trained model for the entire dataset
income_df['predicted_dt'] = dt_classifier.predict(X)
'''
# Calculate and Print the Overall False Positive Rate for income dataset
tn = (income_df['income'] == 0) & (income_df['predicted_dt'] == 0)  # True Negatives
fp = (income_df['income'] == 0) & (income_df['predicted_dt'] == 1)  # False Positives
overall_fp_rate = fp.sum() / (fp.sum() + tn.sum())
print("Overall False Positive Rate of the Dynamic Decision Tree Classifier:", overall_fp_rate)

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
'''
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
plt.title('ROC Curve for the Dynamic Decision Tree Classifier')
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
print("Most 10 divergent itemsets")
print(FP_fm.head(10))


