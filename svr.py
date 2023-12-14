import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from mlxtend.plotting import heatmap
import matplotlib.pyplot as plt

# Load your dataset
file_path = r'C:\Users\Pragna\OneDrive\Desktop\ML_Hourly_data\multiple regression\regression\BTM_hourlyy.csv'
df = pd.read_csv(file_path)
# Assuming df is your DataFrame
df = df.apply(pd.to_numeric, errors='coerce')
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

if not non_numeric_columns.empty:
    print("Columns with non-numeric values:", non_numeric_columns)
else:
    print("No columns with non-numeric values.")
# You may need to load your data into a DataFrame if it's not already in one.
print("total null values")
print(df.isnull().sum().sum())
print("Infinite values:", np.any(np.isinf(df)))
print("NaN values:", np.any(np.isnan(df)))
print("Before imputation:\n", df.head())
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print("After imputation:\n", df_imputed.head())

# Select features and target variable
X = df_imputed.drop("PM2.5", axis=1)
y = df_imputed["PM2.5"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVR model
svr_model = SVR(kernel='rbf')  # You can experiment with different kernels like 'linear', 'poly', etc.
svr_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svr_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
