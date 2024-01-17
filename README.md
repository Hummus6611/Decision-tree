# Decision-tree
This Python script utilizes the Scikit-learn library to build, fine-tune, and evaluate a Decision Tree classifier for predicting iPhone purchases based on a given dataset. This script serves as a comprehensive pipeline for developing and assessing the performance of the Decision Tree classifier.

Here's a breakdown of the script:

Data Preparation:
Loads the training dataset from a CSV file.
Splits the data into features (X) and target labels (y).
Divides the data into training and validation sets, preserving the order without randomization.
Extracts the 'Index' column from the validation set for association with predictions.

Decision Tree Model:
Creates a Decision Tree classifier.

Hyperparameter Tuning:
Defines a set of hyperparameters to search through (max_depth and min_samples_leaf).
Conducts a grid search using cross-validation to find the best hyperparameters.

Training:
Trains a Decision Tree model using the best hyperparameters obtained from the grid search.

Evaluation:
Makes predictions on the validation set.

Report Model Performance:
Calculates and prints accuracy and F1-score of the model on the validation set.

Results Presentation:
Creates a DataFrame with the updated 'Index' and the model's predictions.
Saves the validation results to a CSV file named 'validation_results.csv'.
