import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data (Month vs Sales)
months = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
sales = np.array([100, 120, 130, 150, 170])

# Model
model = LinearRegression()
model.fit(months, sales)

# Predict next 3 months
future_months = np.array([6, 7, 8]).reshape(-1, 1)
predicted_sales = model.predict(future_months)

print("Predicted Sales:", predicted_sales)

# Plot
plt.scatter(months, sales, color='blue', label='Actual')
plt.plot(months, model.predict(months), color='red', label='Regression Line')
plt.scatter(future_months, predicted_sales, color='green', label='Predicted')
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()


# 2 forecasting process
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
months = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
sales = np.array([100, 120, 130, 150, 170])

# Model
model = LinearRegression()
model.fit(months, sales)

# Predict
predicted = model.predict(months)

# Accuracy Calculation
accuracy = 100 - (np.mean(np.abs((sales - predicted) / sales)) * 100)

print("Actual:", sales)
print("Predicted:", predicted)
print("Accuracy:", accuracy, "%")

#3 correlation and regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data (Advertising vs Sales)
advertising = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
sales = np.array([100, 150, 200, 250, 300])

# Correlation
correlation = np.corrcoef(advertising.flatten(), sales)[0, 1]
print("Correlation:", correlation)

# Regression
model = LinearRegression()
model.fit(advertising, sales)

# Prediction
new_ad = np.array([[35]])
predicted_sales = model.predict(new_ad)
print("Predicted Sales for 35k advertising:", predicted_sales)

# Plot
plt.scatter(advertising, sales)
plt.plot(advertising, model.predict(advertising), color='red')
plt.xlabel("Advertising")
plt.ylabel("Sales")
plt.show()

# 4 tsa in forecasting
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Data
months = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
sales = np.array([100, 120, 140, 160, 180])

# Plot time series
plt.plot(months, sales, marker='o')

# Trend line
model = LinearRegression()
model.fit(months, sales)
trend = model.predict(months)

plt.plot(months, trend, color='red')
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Time Series Analysis")
plt.show()

#5 ES & ADVANCE TECHNIQUE IN FORECASTING
import numpy as np

# Data
sales = [100, 120, 130, 150, 170]
alpha = 0.5

# Exponential Smoothing
forecast = [sales[0]]

for i in range(1, len(sales)):
    new_value = alpha * sales[i-1] + (1 - alpha) * forecast[i-1]
    forecast.append(new_value)

print("Forecast values:", forecast)

#6 Sales & Profit Forecast
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
months = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
sales = np.array([100, 120, 140, 160, 180])

# Model
model = LinearRegression()
model.fit(months, sales)

# Predict month 6
predicted_sales = model.predict([[6]])[0]

# Profit Calculation
fixed_cost = 50
variable_cost = 0.3 * predicted_sales
profit = predicted_sales - (fixed_cost + variable_cost)

print("Predicted Sales:", predicted_sales)
print("Profit:", profit)

#7 Qualitative Forecasting
import numpy as np

# Expert opinions
opinions = [100, 110, 105, 115]

# Average forecast
forecast = np.mean(opinions)

print("Forecast Demand:", forecast)