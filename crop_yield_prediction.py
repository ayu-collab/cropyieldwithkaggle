import pandas as pd

# Load the dataset
data = pd.read_csv('Crop_Yield_Prediction.csv')  

# Print the first few rows
print(data.head())


# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

print("Columns in the dataset:\n", data.columns)

# Assuming 'yield' is the column you want to predict
# Select the relevant features and target variable
X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
y = data['Yield']

print("Features (X):\n", X.head())
print("Target (y):\n", y.head())

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)

#print("Predicted yields for test data:", y_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Yields')
plt.ylabel('Predicted Yields')
plt.title('Actual vs. Predicted Crop Yields')
plt.show()






