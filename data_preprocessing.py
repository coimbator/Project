import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your dataset
file_path = 'data/HAR.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Data Quality Checks and Cleaning
# Replace missing values (NaN) with the mean value for each column
#data.fillna(data.mean(), inplace=True)

# Apply Label Encoding to the 'Activity' column
label_encoders = {}
if 'Activity' in data.columns:
    le = LabelEncoder()
    data['Activity'] = le.fit_transform(data['Activity'])
    label_encoders['Activity'] = le

# Save the cleaned data to a new CSV file
cleaned_file_path = 'data/cleaned_dataset.csv'
data.to_csv(cleaned_file_path, index=False)

# Standardize the dataset
scaler = StandardScaler()
X = data.drop(columns=['Activity']).values
X = scaler.fit_transform(X)

# Save the standardized features and target to separate files
standardized_features_path = 'data/standardized_features.csv'
pd.DataFrame(X).to_csv(standardized_features_path, index=False)
target_path = 'data/target.csv'
data['Activity'].to_csv(target_path, index=False)

print("Data preprocessing completed. Files saved:")
print("- Cleaned dataset: cleaned_dataset.csv")
print("- Standardized features: standardized_features.csv")
print("- Target: target.csv")

import os
os.system(f"git add {cleaned_file_path} {standardized_features_path} {target_path}")
os.system('git commit -m "Add preprocessed and standardized data files"')
os.system('git push')

print("Data preprocessing completed. Files saved and pushed to the repository:")
print(f"- Cleaned dataset: {cleaned_file_path}")
print(f"- Standardized features: {standardized_features_path}")
print(f"- Target: {target_path}")
