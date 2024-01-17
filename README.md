# Decision-tree
This Python script utilizes the Scikit-learn library to build, fine-tune, and evaluate a Decision Tree classifier for predicting iPhone purchases based on a given dataset. This script serves as a comprehensive pipeline for developing and assessing the performance of the Decision Tree classifier.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Step 1: Data Preparation
# Load the training dataset
data = pd.read_csv('/kaggle/input/cs770/train_data.csv')

# Split data into features (X) and target labels (y)
X = data[['Index', 'Gender', 'Age', 'Salary']]
y = data['Purchase Iphone']

# Calculate the number of samples for the validation set (20% of the data)
validation_size = int((0.2 * len(data))+1)

# Split the data into training and validation sets without randomization
X_train, y_train = X.iloc[:len(X) - validation_size], y.iloc[:len(y) - validation_size]
X_val, y_val = X.iloc[len(X) - validation_size:], y.iloc[len(y) - validation_size:]

# Extract the Index column for associating with predictions
X_val_index = X_val['Index']

# Drop the Index column before training
X_train = X_train.drop('Index', axis=1)
X_val = X_val.drop('Index', axis=1)

# Step 2: Decision Tree Model
# Create a Decision Tree classifier
clf = DecisionTreeClassifier()

# Step 3: Hyperparameter Tuning
# Define the hyperparameters to search
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Step 4: Training
# Train the Decision Tree model with the best hyperparameters
best_clf = DecisionTreeClassifier(**best_params)
best_clf.fit(X_train, y_train)

# Step 5: Evaluation
# Make predictions on the validation set
y_pred = best_clf.predict(X_val)

# Step 6: Report Model Performance
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")

# Create the DataFrame with the updated index
validation_results = pd.DataFrame({'Index': range(1, 61), 'Purchase iPhone Prediction': y_pred})

# Save the validation results to a CSV file
validation_results.to_csv('validation_results.csv', index=False)
