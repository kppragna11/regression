import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load the data
file_path = r'C:\Users\Pragna\OneDrive\Desktop\ML_Hourly_data\multiple regression\regression\BTM_hourly keerthana.csv'
data = pd.read_csv(file_path)

# Assuming 'PM2.5' is the dependent variable
X = data.drop(['PM2.5'], axis=1)
y = data['PM2.5']
non_numeric_columns = data.columns[data.dtypes == 'object']
for col in non_numeric_columns:
    non_numeric_values = data[col][pd.to_numeric(data[col], errors='coerce').isnull()]
    print(f'Column {col} has non-numeric values: {non_numeric_values.unique()}')
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
# Replace non-numeric values in 'y' with a default value (e.g., 0)
y = pd.to_numeric(y, errors='coerce').fillna(0)

# Separate numeric and non-numeric columns
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
X_numeric = X.select_dtypes(include=[np.number])
X_non_numeric = X[non_numeric_columns]
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
# Impute missing values
imputer_numeric = SimpleImputer(strategy='mean')
imputer_non_numeric = SimpleImputer(strategy='most_frequent')
X_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(X_numeric), columns=X_numeric.columns)
X_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(X_non_numeric), columns=X_non_numeric.columns)
X_imputed = pd.concat([X_numeric_imputed, X_non_numeric_imputed], axis=1)

# Convert all columns to numeric (assuming it's safe to do so)
X_imputed = X_imputed.apply(pd.to_numeric, errors='coerce')

# Handle outliers (optional)
Q1 = X_imputed.quantile(0.25)
Q3 = X_imputed.quantile(0.75)
IQR = Q3 - Q1
X_no_outliers = X_imputed[~((X_imputed < (Q1 - 1.5 * IQR)) | (X_imputed > (Q3 + 1.5 * IQR))).any(axis=1)]

# Separate features (X) and target variable (y) after handling outliers
X_no_outliers = X_no_outliers.apply(pd.to_numeric, errors='coerce')  # Ensure numeric types
X = X_no_outliers  # Rename for clarity
y = data.loc[X.index, 'PM2.5']
# Replace non-numeric values in 'y' with a default value (e.g., 0)
y = pd.to_numeric(y, errors='coerce').fillna(0)

# Convert 'y' to numeric (assuming it's safe to do so)
y = pd.to_numeric(y, errors='coerce')

# Separate features (X) and target variable (y) after handling outliers
X_no_outliers = X_no_outliers.apply(pd.to_numeric, errors='coerce')  # Ensure numeric types
X = X_no_outliers  # Rename for clarity
y = data.loc[X.index, 'PM2.5']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Perform multiple regression using gradient descent with optimal learning rate
def gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)  # Initialize weights
    for iteration in range(num_iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = 2/m * np.dot(X.T, errors)
        theta -= learning_rate * gradient
    return theta

# Search for the optimal learning rate
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1]
min_mse = float('inf')
optimal_learning_rate = None
optimal_predictions = None

for rate in learning_rates:
    theta = gradient_descent(X_train_scaled.values, y_train.values, learning_rate=rate, num_iterations=1000)
    predictions = np.dot(X_test_scaled.values, theta)
    # Calculate MSE
    non_nan_indices = ~np.isnan(y_test) & ~np.isnan(predictions)
    y_test_non_nan = y_test[non_nan_indices]
    predictions_non_nan = predictions[non_nan_indices]

    mse = mean_squared_error(y_test_non_nan, predictions_non_nan)

    
    print(f"Learning Rate: {rate}, Mean Squared Error: {mse}")
    
    # Update optimal learning rate if current MSE is lower
    if mse < min_mse:
        min_mse = mse
        optimal_learning_rate = rate
        optimal_predictions = predictions

# Train the model with the optimal learning rate
optimal_theta = gradient_descent(X_train_scaled.values, y_train.values, learning_rate=optimal_learning_rate, num_iterations=1000)

# Make predictions on the test set
predictions_test = np.dot(X_test_scaled.values, optimal_theta)

# Calculate errors
errors_test = predictions_test - y_test

# Print actual and predicted values
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_test})
print(result_df)

print(f"\nOptimal Learning Rate: {optimal_learning_rate}")
print(f"Mean Squared Error on Test Set: {min_mse}")
