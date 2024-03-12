from sklearn.base import BaseEstimator


def score(text: str, model: BaseEstimator, threshold: float) -> tuple[bool, float]:

    # Get the propensity score
    propensity = model.predict_proba([text])[0][1]

    # Make a prediction based on the threshold
    prediction = propensity > threshold

    return prediction.item(), propensity
