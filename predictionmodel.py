import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('sample.csv')

# Convert 'Transaction Date' to datetime
data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])

# Feature Engineering
data['Month'] = data['Transaction Date'].dt.month
data['Year'] = data['Transaction Date'].dt.year

# Group by Product ID, Product Name, Category, and Date to get monthly sales
monthly_sales = data.groupby(['Product ID', 'Product Name', 'Category', 'Year', 'Month']).agg({
    'Total Amount': 'sum'
}).reset_index()

# Prepare features and target
X = monthly_sales[['Year', 'Month']]
y = monthly_sales['Total Amount']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
monthly_sales['Predicted Sales'] = model.predict(X[['Year', 'Month']])

# Save to CSV
monthly_sales.to_csv('predicted_sales.csv', index=False)
