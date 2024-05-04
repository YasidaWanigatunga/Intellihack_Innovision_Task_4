import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = 'Watches Bags Accessories.csv'
df = pd.read_csv(file_path)

# Check for null values in the dataset
print("Null Values Check:")
null_values = df.isnull().sum()
print(null_values[null_values > 0])

# Check for unique values to identify potential data errors in categorical columns
print("\nUnique Values Check:")
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    unique_values = df[col].unique()
    print(f"Column: {col}, Unique Values: {len(unique_values)}")
    if len(unique_values) < 20:
        print(unique_values)

# Check for outliers in numerical columns using interquartile range (IQR)
print("\nOutlier Check:")
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column: {col}, Outliers Count: {outliers.shape[0]}")

# 1. Calculate the discount percentage
df['discount_percentage'] = ((df['original_price'] - df['current_price']) / df['original_price']) * 100

# 2. Convert ratings to numerical, assuming ratings are in a categorical format
if df['rating'].dtype == 'object':
    df['rating'] = df['rating'].str.extract('(\d+\.\d+)', expand=False).astype(float)

# 3. Extracting numerical columns and scaling them
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# 4. Label encoding categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Storing the encoder for inverse transform later if needed

# 5. Example of using domain knowledge: classifying products based on price range
def classify_price_range(price):
    if price < 50:
        return 'low'
    elif price < 150:
        return 'medium'
    else:
        return 'high'

df['price_range'] = df['current_price'].apply(classify_price_range)

# Assume there's a 'date' column that holds the date data
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# For ARIMA analysis
target = 'sales'  # Replace with the actual target variable name for time series
y = df[target]

# Plot autocorrelation
autocorrelation_plot(y)
plt.show()

# Train ARIMA model
arima_model = sm.tsa.ARIMA(y, order=(5, 1, 0))  # Adjust the order as needed
arima_model_fit = arima_model.fit(disp=0)

# Forecast the future trends
forecast, stderr, conf_int = arima_model_fit.forecast(steps=10)
print(forecast)

# Linear regression analysis
X = df.drop(columns=[target])
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on test data
y_pred = regressor.predict(X_test)

# Evaluate the model using MAE, MSE, and R-squared
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Plot Actual vs Predicted Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.show()

# Analyze the influence of ratings
plt.figure(figsize=(10, 6))
plt.scatter(X_test['rating'], y_pred, alpha=0.6, label='Predicted Sales')
plt.scatter(X_test['rating'], y_test, alpha=0.6, label='Actual Sales')
plt.xlabel('Rating')
plt.ylabel('Sales')
plt.title('Sales vs. Rating')
plt.legend()
plt.show()

# Create a new dataset for future predictions
future_X = np.linspace(X_test.min(), X_test.max(), 10)
future_pred = regressor.predict(future_X)

# Plot future predictions
plt.figure(figsize=(10, 6))
plt.plot(future_X, future_pred, marker='o', linestyle='-', color='blue', label='Future Predictions')
plt.xlabel('Feature Range')
plt.ylabel('Predicted Sales')
plt.title('Future Sales Predictions')
plt.legend()
plt.show()
