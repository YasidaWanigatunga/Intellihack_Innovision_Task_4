import pandas as pd

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