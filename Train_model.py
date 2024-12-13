import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
#import joblib
import argparse
import pickle

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

# Save the model
#joblib.dump(clf, 'har_model.pkl')

with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)