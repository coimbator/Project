import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import mlflow
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata1", type=str, required=True, help='Dataset for training')
parser.add_argument("--trainingdata2", type=str, required=True, help='Dataset for training')
parser.add_argument("--testingdata1", type=str, required=True, help='Dataset for testing')
parser.add_argument("--testingdata2", type=str, required=True, help='Dataset for testing')
args = parser.parse_args()
mlflow.autolog()
# Load preprocessed data
X_train= pd.read_csv(args.trainingdata1).values
y_train = pd.read_csv(args.trainingdata2).values.ravel()
X_test= pd.read_csv(args.testingdata1).values
y_test= pd.read_csv(args.testingdata2).values.ravel()

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

# Save the model
joblib.dump(clf, 'har_model.pkl')