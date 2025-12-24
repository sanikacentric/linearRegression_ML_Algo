#Linear Regression – Predicting Ride Price (Uber / Ola)
#Real-time problem

#Predict the price of a ride based on:

#Distance in km

#Estimated time

#Surge factor

#This is a regression problem (predict a number). Target and features are mapping to each other

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Each row: [distance_km, duration_minutes, surge_multiplier]
X = np.array([
    [2.0,  10, 1.0],
    [5.0,  20, 1.2],
    [1.0,   5, 1.0],
    [10.0, 30, 1.5],
    [7.0,  25, 1.3],
    [3.0,  12, 1.0],
])

# price in dollars
y = np.array([5, 12, 3, 25, 18, 7])

# 🔹 1. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 2. Create and train the model on TRAIN data only
model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 🔹 3. Evaluate on TEST data
y_pred_test = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print("Test MSE:", mse)

# 🔹 4. Predict for a new trip (unseen data)
new_trip = np.array([[6.0, 22, 1.2]])
pred_price = model.predict(new_trip)
print("Predicted price for new trip:", pred_price[0])

#How to explain this in interview (simple)



#First, I split the data into train and test sets using train_test_split.
#The model learns patterns only from the train set.
#Then I use the test set (data the model has never seen) to check how well it generalizes.
#I measure this using an error metric like Mean Squared Error (MSE).#

