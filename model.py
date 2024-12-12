import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()
mlflow.autolog()

df = pd.read_csv(args.trainingdata)
print(df)

X = df.drop(columns=['Activity', 'subject']).values  # Drops the Activity and Subject column
Y = df['Activity'].values  # Extract the label column

# Check the number of rows
print(f"Number of rows: {len(X)}")

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)