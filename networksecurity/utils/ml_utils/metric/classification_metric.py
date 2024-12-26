from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score,precision_score,recall_score
import sys


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculates classification metrics (F1 score, recall, and precision) for a given set of true and predicted values.

    Parameters:
        y_true (array-like): True labels (ground truth).
        y_pred (array-like): Predicted labels from the model.

    Returns:
        ClassificationMetricArtifact: An object containing the F1 score, precision, and recall.
    """
    try:
        # Calculate the F1 score: harmonic mean of precision and recall
        model_f1_score = f1_score(y_true, y_pred)

        # Calculate recall: proportion of true positives identified correctly
        model_recall_score = recall_score(y_true, y_pred)

        # Calculate precision: proportion of predicted positives that are true positives
        model_precision_score = precision_score(y_true, y_pred)

        # Create a classification metric artifact object to encapsulate the metrics
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score, 
            recall_score=model_recall_score
        )

        # Return the classification metric artifact
        return classification_metric
    except Exception as e:
        # Raise a custom exception in case of errors
        raise NetworkSecurityException(e, sys)
