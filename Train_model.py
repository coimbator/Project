import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import mlflow
import joblib

mlflow.autolog()
# Load preprocessed data
X_train= pd.read_csv('data/standardized_train_features.csv').values
y_train = pd.read_csv('data/train_target.csv').values.ravel()
X_test= pd.read_csv('data/standardized_test_features.csv').values
y_test= pd.read_csv('data/test_target.csv').values.ravel()

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