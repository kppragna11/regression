import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Read the csv file
file_path = r'C:\Users\Pragna\OneDrive\Desktop\ML_Hourly_data\multiple regression\regression\BTM_hourlyy.csv'
df= pd.read_csv(file_path)

non_numeric_columns = df.columns[df.dtypes == 'object']
for col in non_numeric_columns:
    non_numeric_values = df[col][pd.to_numeric(df[col], errors='coerce').isnull()]
    print(f'Column {col} has non-numeric values: {non_numeric_values.unique()}')
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
print(df.describe())
print(df.corr())
# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
lower_threshold = df_imputed['PM2.5'].quantile(0.05)
upper_threshold = df_imputed['PM2.5'].quantile(0.95)
df_imputed['PM2.5'] = np.where(df_imputed['PM2.5'] > upper_threshold, upper_threshold, df_imputed['PM2.5'])
df_imputed['PM2.5'] = np.where(df_imputed['PM2.5'] < lower_threshold, lower_threshold, df_imputed['PM2.5'])
x = df_imputed.drop('PM2.5', axis=1)
y = df_imputed['PM2.5']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)


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
optimal_theta = None

for rate in learning_rates:
    theta = gradient_descent(x_train_std, y_train, learning_rate=rate, num_iterations=1000)
    predictions = np.dot(x_test_std, theta)
    # Calculate MSE
    non_nan_indices = ~np.isnan(y_test) & ~np.isnan(predictions)
    y_test_non_nan = y_test[non_nan_indices]
    predictions_non_nan = predictions[non_nan_indices]

    if len(y_test_non_nan) > 0:
        mse = mean_squared_error(y_test_non_nan, predictions_non_nan)
        print(f"Learning Rate: {rate}, Mean Squared Error: {mse}")

    # Update optimal learning rate if current MSE is lower
    if mse < min_mse:
        min_mse = mse
        optimal_learning_rate = rate
        optimal_predictions = predictions
        optimal_theta = theta

# Train the model with the optimal learning rate on the entire dataset
optimal_theta_train = gradient_descent(x_train_std, y_train, learning_rate=optimal_learning_rate, num_iterations=1000)

# Make predictions on the training set
predictions_train = np.dot(x_train_std, optimal_theta_train)

# Calculate RMSE on training set
rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))
print(f"Root Mean Squared Error on Training Set: {rmse_train}")

# Calculate R-squared value on training set
r2_train = r2_score(y_train, predictions_train)
print(f"R-squared Value on Training Set: {r2_train}")

# Calculate accuracy (you need to define a threshold for binary classification)
# For regression tasks, accuracy might not be a suitable metric
# This is just an example; you might need a different metric for your specific problem
threshold = 0.5
accuracy_train = accuracy_score(y_train > threshold, predictions_train > threshold)
print(f"Accuracy on Training Set: {accuracy_train}")
# Additional: Check for multicollinearity using Variance Inflation Factor (VIF)
def calculate_vif(data):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the dataset.

    Parameters:
    - data: pandas DataFrame or NumPy array

    Returns:
    - vif_data: DataFrame containing VIF values for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Check if the input is a DataFrame or NumPy array
    if isinstance(data, pd.DataFrame):
        vif_data = pd.DataFrame()
        vif_data["Variable"] = data.columns
        vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    elif isinstance(data, np.ndarray):
        vif_data = pd.DataFrame()
        vif_data["Variable"] = [f"Var_{i}" for i in range(data.shape[1])]
        vif_data["VIF"] = [variance_inflation_factor(data, i) for i in range(data.shape[1])]
    else:
        raise ValueError("Unsupported data type. Please provide either a pandas DataFrame or a NumPy array.")

    return vif_data

# Example usage:
vif_results = calculate_vif(x_train_std)
print(vif_results)

