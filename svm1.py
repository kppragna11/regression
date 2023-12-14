import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load your data
file_path = r'C:\Users\Pragna\OneDrive\Desktop\ML_Hourly_data\multiple regression\regression\BTM_hourlyy.csv'
df = pd.read_csv(file_path)

# Handle non-numeric values
df = df.apply(pd.to_numeric, errors='coerce')
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

if not non_numeric_columns.empty:
    print("Columns with non-numeric values:", non_numeric_columns)
else:
    print("No columns with non-numeric values.")

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Select features and target variable
X = df_imputed.drop("PM2.5", axis=1)
y = df_imputed["PM2.5"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf'],
              'gamma': [0.1, 1, 'scale']}

# Create the SVM regressor
svm_regressor = SVR()

# Instantiate GridSearchCV with parallel processing
grid_search = GridSearchCV(estimator=svm_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model to the entire dataset with feature scaling
grid_search.fit(X, y)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_svm = grid_search.best_estimator_

# Make predictions on the test set
y_pred_best = best_svm.predict(X_test)

# Evaluate the best model
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)

print(f"Mean Squared Error with Best Model: {mse_best}")
print(f"Root Mean Squared Error (RMSE) with Best Model: {rmse_best}")



