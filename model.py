import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()
mlflow.autolog()

df = pd.read_csv(args.trainingdata)
print(df)

def filter_columns(data):
    # Exclude the label column and use all other columns as inputs
    X = data.drop(columns=['Activity', 'subject']).values  # Drops the 'type' column
    return X

# Assuming df is your DataFrame
X = filter_columns(df)
Y = df['Activity'].values  # Extract the label column

# Check the number of rows
print(f"Number of rows: {len(X)}")

#Split the data and keep 20% back for testing later
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
print("Train length", len(X_train))
print("Test length", len(X_test))

models = [
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=42),
        (X_train, Y_train),
        (X_test, Y_test)
    ),
    (
        "Logistic Regression",
        LogisticRegression(),
        (X_train, Y_train),
        (X_test, Y_test)
    ),
    (
        "SVM",
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        (X_train, Y_train),
        (X_test, Y_test)
    )
]

reports = []

for model_name, model, train_set, test_set in models:
    X_train = train_set[0]
    Y_train = train_set[1]
    X_test = test_set[0]
    Y_test = test_set[1]

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    report = classification_report(Y_test, y_pred, output_dict=True)
    reports.append(report)
    print(reports)