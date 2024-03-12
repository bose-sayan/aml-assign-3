import unittest
import joblib
from score import score


class TestScoreFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the trained model
        cls.model = joblib.load(
            "/home/sbose/AML/assign-3/problem-1/model/spam_detection_model.pkl"
        )

    def test_smoke(self):
        # Smoke test: just check if the function produces some output without crashing
        text = "Sample text"
        prediction, propensity = score(text, self.model, 0.5)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(propensity)

    def test_formats(self):
        # Format test: check if the output types are as expected
        text = "Sample text"
        prediction, propensity = score(text, self.model, 0.5)
        print(type(prediction))
        self.assertIsInstance(prediction, bool)
        self.assertIsInstance(propensity, float)

    def test_prediction_range(self):
        # Check if prediction value is 0 or 1
        text = "Sample text"
        prediction, _ = score(text, self.model, 0.5)
        self.assertIn(prediction, [True, False])

    def test_propensity_range(self):
        # Check if propensity score is between 0 and 1
        text = "Sample text"
        _, propensity = score(text, self.model, 0.5)
        self.assertGreaterEqual(propensity, 0)
        self.assertLessEqual(propensity, 1)

    def test_threshold_zero(self):
        # Check if the prediction is always 1 (True) when threshold is 0
        text = "Sample text"
        prediction, _ = score(text, self.model, 0)
        self.assertTrue(prediction)

    def test_threshold_one(self):
        # Check if the prediction is always 0 (False) when threshold is 1
        text = "Sample text"
        prediction, _ = score(text, self.model, 1)
        self.assertFalse(prediction)

    def test_spam_prediction(self):
        # Check if an obvious spam input text results in prediction 1 (True)
        spam_text = "Win a free iPhone! Click here!"
        prediction, _ = score(spam_text, self.model, 0.5)
        self.assertTrue(prediction)

    def test_non_spam_prediction(self):
        # Check if an obvious non-spam input text results in prediction 0 (False)
        non_spam_text = "Hello, how are you?"
        prediction, _ = score(non_spam_text, self.model, 0.5)
        self.assertFalse(prediction)


if __name__ == "__main__":
    unittest.main()
