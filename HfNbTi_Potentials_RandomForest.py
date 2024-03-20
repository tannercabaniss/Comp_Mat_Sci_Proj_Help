import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean

# Load your data
data_file_path = 'data.txt'
data = pd.read_csv(data_file_path, delimiter='\s+', header=None)

# Ensure data types are float
for i in range(len(data.columns)):
    data.iloc[:, i] = data.iloc[:, i].astype(float)
    
X = data.iloc[:, :2].values  # Features
y = data.iloc[:, 2:].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for random search
param_grid_random = {
    'n_estimators': [2, 3, 4, 5, 10, 50, 100, 200, 300, 500, 1000, 2000, 5000],  # Number of trees
    'max_depth': [None, 1, 2, 3, 5],  # Maximum depth of a tree
    'min_samples_split': [2, 3, 4, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4, 8],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Method of selecting samples for training each tree
}

# Define the Random Forest base model with default hyperparameters
base_model = RandomForestRegressor(random_state=42)

# Define the Random Forest model for Random Search
random_search_model = HalvingRandomSearchCV(estimator=base_model, param_distributions=param_grid_random,
                                         scoring='neg_mean_squared_error', verbose=1, random_state=42, factor=2, cv=2)

# Fit the base model to the training data
base_model.fit(X_train, y_train)

# Fit the model with Random Search
random_search_model.fit(X_train, y_train)

# Predict on the test set using all models
y_pred_base_model = base_model.predict(X_test)
y_pred_random_search_model = random_search_model.best_estimator_.predict(X_test)

train_y_pred_base_model = base_model.predict(X_train)
train_mse_base_model = mean_squared_error(y_train, train_y_pred_base_model)
train_mae_base_model = mean_absolute_error(y_train, train_y_pred_base_model)
train_rmse_base_model = np.sqrt(train_mse_base_model)
train_r2_base_model = r2_score(y_train, train_y_pred_base_model)

# Calculate evaluation metrics for the model with Random Search on the training set
train_y_pred_random_search_model = random_search_model.predict(X_train)
train_mse_random_search_model = mean_squared_error(y_train, train_y_pred_random_search_model)
train_mae_random_search_model = mean_absolute_error(y_train, train_y_pred_random_search_model)
train_rmse_random_search_model = np.sqrt(train_mse_random_search_model)
train_r2_random_search_model = r2_score(y_train, train_y_pred_random_search_model)

# Calculate evaluation metrics for the base model
mse_base_model = mean_squared_error(y_test, y_pred_base_model)
mae_base_model = mean_absolute_error(y_test, y_pred_base_model)
rmse_base_model = np.sqrt(mse_base_model)
r2_base_model = r2_score(y_test, y_pred_base_model)

# Calculate evaluation metrics for the model with Random Search
mse_random_search_model = mean_squared_error(y_test, y_pred_random_search_model)
mae_random_search_model = mean_absolute_error(y_test, y_pred_random_search_model)
rmse_random_search_model = np.sqrt(mse_random_search_model)
r2_random_search_model = r2_score(y_test, y_pred_random_search_model)

train_metrics_table = [
    ['Base Model', train_mse_base_model, train_mae_base_model, train_rmse_base_model, train_r2_base_model],
    ['Random Search Model with Halving CV', train_mse_random_search_model, train_mae_random_search_model, train_rmse_random_search_model, train_r2_random_search_model]
]

# Define test evaluation metrics in a list of lists
test_metrics_table = [
    ['Base Model', mse_base_model, mae_base_model, rmse_base_model, r2_base_model],
    ['Random Search Model with Halving CV', mse_random_search_model, mae_random_search_model, rmse_random_search_model, r2_random_search_model]
]

train_metrics_table_rounded = [[name, round(mse, 5), round(mae, 5), round(rmse, 5), round(r2, 5)] for name, mse, mae, rmse, r2 in train_metrics_table]

# Define test evaluation metrics in a list of lists, rounding each value to 5 decimal places
test_metrics_table_rounded = [[name, round(mse, 5), round(mae, 5), round(rmse, 5), round(r2, 5)] for name, mse, mae, rmse, r2 in test_metrics_table]

# Define headers for the table
headers = ["Model", "Mean Squared Error", "Mean Absolute Error", "Root Mean Squared Error", "R-squared"]

# Print training evaluation metrics table
print()
print("Training Set Evaluation Metrics:")
print(tabulate(train_metrics_table, headers=headers, tablefmt="fancy_grid"))

# Print test evaluation metrics table
print("\nTest Set Evaluation Metrics:")
print(tabulate(test_metrics_table, headers=headers, tablefmt="fancy_grid"))
# Print the best hyperparameters found by Random Search
print("Best Hyperparameters - Random Search:", random_search_model.best_params_)

def objective_function_inverse(x, model):
    return np.square(model.predict([x])[0] - 0.333).sum()

# Define the desired target output
desired_output = np.array([0.333, 0.333, 0.333])

# Calculate the distance between each data point's output and the desired target output
distances = [euclidean(y_i, desired_output) for y_i in y_train]

# Find the index of the data point with the smallest distance
closest_index = np.argmin(distances)

# Use the input values of the closest data point as the initial guess for inverse prediction
initial_guess_inverse = X_train[closest_index]

# Use scipy's minimize function to find the input values that minimize the objective function
result_inverse_base = minimize(objective_function_inverse, initial_guess_inverse, args=(base_model,), method='Nelder-Mead')

# Get the optimized input values
optimized_input_values_inverse_base = result_inverse_base.x

# Print optimized input values for inverse prediction
print("Optimized Input Values for Inverse Prediction (Base):", optimized_input_values_inverse_base)

result_inverse_random = minimize(objective_function_inverse, initial_guess_inverse, args=(random_search_model.best_estimator_,), method='Nelder-Mead')

# Get the optimized input values
optimized_input_values_inverse_random = result_inverse_random.x

# Print optimized input values for inverse prediction
print("Optimized Input Values for Inverse Prediction (Random):", optimized_input_values_inverse_random)