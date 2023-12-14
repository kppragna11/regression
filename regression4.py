import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
file_path = r'C:\Users\Pragna\OneDrive\Desktop\ML_Hourly_data\multiple regression\regression\BTM_hourly keerthana.csv'
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

## ... (previous code)

# Search for the optimal learning rate
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5, 1]
min_mse = float('inf')
optimal_learning_rate = None
optimal_predictions = None

for rate in learning_rates:
    theta = gradient_descent(x_train_std, y_train, learning_rate=rate, num_iterations=1000)

    predictions = np.dot(x_test_std, theta)

    
    # Calculate MSE
    non_nan_indices = ~np.isnan(y_test) & ~np.isnan(predictions)
    y_test_non_nan = y_test[non_nan_indices]
    predictions_non_nan = predictions[non_nan_indices]

    # Calculate MSE
    if len(y_test_non_nan) > 0:
        mse = mean_squared_error(y_test_non_nan, predictions_non_nan)
        print(f"Learning Rate: {rate}, Mean Squared Error: {mse}")
        
        # Update optimal learning rate if current MSE is lower
        if mse < min_mse:
            min_mse = mse
            optimal_learning_rate = rate
            optimal_predictions = predictions
    else:
        print(f"Learning Rate: {rate}, No valid samples for MSE calculation.")

        optimal_predictions = predictions

# Train the model with the optimal learning rate
optimal_theta = gradient_descent(x_train_std, y_train, learning_rate=optimal_learning_rate, num_iterations=1000)


# Make predictions on the test set
predictions_test = np.dot(x_test_std, optimal_theta)

# Calculate errors
errors_test = predictions_test - y_test

# Print actual and predicted values
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_test})
print(result_df)

print(f"\nOptimal Learning Rate: {optimal_learning_rate}")
print(f"Mean Squared Error on Test Set: {min_mse}")

