pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
url = 'https://www.kaggle.com/datasets/niharika41298/air-quality-prediction/download'
data = pd.read_csv(url)

# Display first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Fill missing values (for simplicity, using the mean of each column)
data.fillna(data.mean(), inplace=True)

# Features and target variable
features = ['PM10', 'NO2', 'CO', 'O3', 'temperature', 'humidity', 'windspeed']
target = 'PM2.5'

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# Plotting true vs predicted PM2.5 levels
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([0, 100], [0, 100], color='red', linestyle='--')  # Line of perfect prediction
plt.title('True vs Predicted PM2.5 Levels')
plt.xlabel('True PM2.5')
plt.ylabel('Predicted PM2.5')
plt.show()
