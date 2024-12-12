import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your dataset
file_path = 'HAR.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Data Quality Checks and Cleaning
# Replace missing values (NaN) with the mean value for each column
data.fillna(data.mean(), inplace=True)

# Save the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_dataset.csv'
data.to_csv(cleaned_file_path, index=False)

# Standardize the dataset
scaler = StandardScaler()
X = data.iloc[:, :-1].values
X = scaler.fit_transform(X)

# Save the standardized features and target to separate files
pd.DataFrame(X).to_csv('standardized_features.csv', index=False)
data.iloc[:, -1].to_csv('target.csv', index=False)
