import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset
file_path = 'data/HAR.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Data Quality Checks and Cleaning
# Replace missing values (NaN) with the mean value for each column
#data.fillna(data.mean(), inplace=True)

# Apply Label Encoding to the 'activity' column
label_encoders = {}
if 'activity' in data.columns:
    le = LabelEncoder()
    data['activity'] = le.fit_transform(data['activity'])
    label_encoders['activity'] = le

# Save the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_dataset.csv'
data.to_csv(cleaned_file_path, index=False)

# Standardize the dataset
scaler = StandardScaler()
X = data.drop(columns=['activity']).values
X = scaler.fit_transform(X)

# Save the standardized features and target to separate files
pd.DataFrame(X).to_csv('standardized_features.csv', index=False)
data['activity'].to_csv('target.csv', index=False)

print("Data preprocessing completed. Files saved:")
print("- Cleaned dataset: cleaned_dataset.csv")
print("- Standardized features: standardized_features.csv")
print("- Target: target.csv")
