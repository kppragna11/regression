import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load data from CSV file
file_path = r'C:\Users\Pragna\OneDrive\Desktop\ML_Hourly_data\multiple regression\regression\BTM_hourly keerthana.csv'

df = pd.read_csv(file_path)

# Display basic statistics of the data
non_numeric_columns = df.columns[df.dtypes == 'object']
for col in non_numeric_columns:
    non_numeric_values = df[col][pd.to_numeric(df[col], errors='coerce').isnull()]
    print(f'Column {col} has non-numeric values: {non_numeric_values.unique()}')
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Display basic statistics of the data
print(df.describe())

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Step 2: Data Preprocessing

# Handle missing values (imputation)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Handle outliers (you can use a more sophisticated method based on your data)
# Assuming 'PM2.5' is the target variable
lower_threshold = df_imputed['PM2.5'].quantile(0.05)
upper_threshold = df_imputed['PM2.5'].quantile(0.95)

df_imputed['PM2.5'] = np.where(df_imputed['PM2.5'] > upper_threshold, upper_threshold, df_imputed['PM2.5'])
df_imputed['PM2.5'] = np.where(df_imputed['PM2.5'] < lower_threshold, lower_threshold, df_imputed['PM2.5'])

# Step 3: Split the data into features (X) and target variable (y)
X = df_imputed.drop('PM2.5', axis=1)
y = df_imputed['PM2.5']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build a regression model with a pipeline for preprocessing

# Create a pipeline with preprocessing steps and the regression model
model = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('regression', LinearRegression())  # Linear Regression model
])

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model (you can use other metrics as well)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Additional: Check for multicollinearity using Variance Inflation Factor (VIF)
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

vif_results = calculate_vif(X_train)
print("VIF Results:")
print(vif_results)
