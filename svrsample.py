import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Example data
X = np.array([[1], [2], [3]])  # Assume a single feature for simplicity
y = np.array([3, 4, 5])

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = y.reshape(-1, 1)  # Reshape only if necessary
y = sc_y.fit_transform(y)  # Fit sc_y to the transformed y values

# Fit SVR Model
svr = SVR(kernel='rbf')
svr.fit(X, y.ravel())

# Example prediction
new_data_point = np.array([[4]])
new_data_point = sc_X.transform(new_data_point)

# Ensure svr is defined before calling predict
prediction = svr.predict(new_data_point)

# Inverse Transform using the same scaler instance
prediction = sc_y.inverse_transform(prediction.reshape(-1, 1))

# Visualize Results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red', label='Data Points')  # Use inverse_transform to get the original scale
plt.plot(sc_X.inverse_transform(X), svr.predict(X), color='blue', label='SVR Prediction')
plt.scatter(sc_X.inverse_transform(new_data_point), prediction, color='green', label='New Data Point Prediction', marker='x', s=100)
plt.title('SVR Model')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

