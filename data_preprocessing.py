import pandas as pd
import os
import subprocess
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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
cleaned_file_path = 'data\cleaned_dataset.csv'
data.to_csv(cleaned_file_path, index=False)

# Standardize the dataset
scaler = StandardScaler()
X = data.drop(columns=['Activity']).values
X = scaler.fit_transform(X)
y = data['Activity'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the standardized features and target to separate files
standardized_train_features_path = 'data/standardized_train_features.csv'
standardized_test_features_path = 'data/standardized_test_features.csv'
train_target_path = 'data/train_target.csv'
test_target_path = 'data/test_target.csv'

output_folder="data/"
# Step 3: Save the preprocessed dataset
output_file1 = os.path.join(output_folder, "standardized_train_features.csv")
pd.DataFrame(X_train).to_csv(standardized_train_features_path, index=False)

output_file2 = os.path.join(output_folder, "standardized_test_features.csv")
pd.DataFrame(X_test).to_csv(standardized_test_features_path, index=False)

output_file3 = os.path.join(output_folder, "train_target.csv")
pd.DataFrame(y_train).to_csv(train_target_path, index=False, header=['Activity'])

output_file4 = os.path.join(output_folder, "test_target.csv")
pd.DataFrame(y_test).to_csv(test_target_path, index=False, header=['Activity'])

print("Data preprocessing completed successfully.")


print("Data preprocessing completed. Files saved and pushed to the repository:")
print(f"- Cleaned dataset: {cleaned_file_path}")
print(f"- Standardized train features: {standardized_train_features_path}")
print(f"- Standardized test features: {standardized_test_features_path}")
print(f"- Train target: {train_target_path}")
print(f"- Test target: {test_target_path}")
