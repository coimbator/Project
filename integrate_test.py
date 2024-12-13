import unittest
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TestRandomForestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Paths to your pickle model and test datasets
        cls.model_path = "model.pkl"  # Update this with your model file path
        cls.test_features_path = "data/standardized_test_features.csv"  # Update with your test features file path
        cls.test_labels_path = "data/test_target.csv"  # Update with your test labels file path

        # Load the trained model from pickle
        with open(cls.model_path, "rb") as f:
            cls.model = pickle.load(f)

        # Load test data
        cls.X_test = pd.read_csv(cls.test_features_path)
        cls.Y_test = pd.read_csv(cls.test_labels_path)  

    def test_model_load(self):
        """Test if the model loads correctly."""
        self.assertIsNotNone(self.model, "Loaded model should not be None")

    def test_prediction_shape(self):
        """Test if predictions match the expected number of samples."""
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.Y_test), "Number of predictions should match the number of test samples")

    def test_accuracy(self):
        """Test if model accuracy meets the minimum threshold."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, y_pred)
        self.assertGreater(accuracy, 0.7, "Model accuracy should be greater than 70%")

    def test_confusion_matrix(self):
        """Test if confusion matrix dimensions are correct for 6 classes."""
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.Y_test, y_pred)
        self.assertEqual(cm.shape, (6, 6), "Confusion matrix should be 6x6 for 6 classes")

    def test_classification_report(self):
        """Test classification report generation."""
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.Y_test, y_pred, output_dict=True)
        self.assertIn("accuracy", report, "Classification report should include accuracy")
        self.assertEqual(len(report) - 3, 6, "Report should have 6 classes plus macro/micro averages")

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"])
