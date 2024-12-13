import pandas as pd
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
cleaned_file_path = 'data/cleaned_dataset.csv'
data.to_csv(cleaned_file_path, index=False)

# Standardize the dataset
scaler = StandardScaler()
X = data.drop(columns=['Activity']).values
X = scaler.fit_transform(X)
y = data['Activity'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the standardized features and target to separate files
standardized_train_features_path = 'standardized_train_features.csv'
standardized_test_features_path = 'standardized_test_features.csv'
train_target_path = 'train_target.csv'
test_target_path = 'test_target.csv'

pd.DataFrame(X_train).to_csv(standardized_train_features_path, index=False)
pd.DataFrame(X_test).to_csv(standardized_test_features_path, index=False)
pd.DataFrame(y_train).to_csv(train_target_path, index=False, header=['Activity'])
pd.DataFrame(y_test).to_csv(test_target_path, index=False, header=['Activity'])

# Save files back to the repository
import os
os.system(f"git add {cleaned_file_path} {standardized_train_features_path} {standardized_test_features_path} {train_target_path} {test_target_path}")
os.system('git commit -m "Add preprocessed, split, and standardized data files"')
os.system('git push')

print("Data preprocessing completed. Files saved and pushed to the repository:")
print(f"- Cleaned dataset: {cleaned_file_path}")
print(f"- Standardized train features: {standardized_train_features_path}")
print(f"- Standardized test features: {standardized_test_features_path}")
print(f"- Train target: {train_target_path}")
print(f"- Test target: {test_target_path}")
