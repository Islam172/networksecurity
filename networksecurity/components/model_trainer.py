import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='ielmaaroufi4', repo_name='networksecurity', mlflow=True)

# Set environment variables for MLflow tracking
#os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/krishnaik06/networksecurity.mlflow"
#os.environ["MLFLOW_TRACKING_USERNAME"] = "krishnaik06"
#os.environ["MLFLOW_TRACKING_PASSWORD"] = "7104284f1bb44ece21e0e2adb4e36a250ae3251f"


class ModelTrainer:
    """
    Handles the training, evaluation, and logging of machine learning models.
    """
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Initializes the ModelTrainer class with required configurations and artifacts.
        
        Parameters:
            model_trainer_config (ModelTrainerConfig): Configuration for model training.
            data_transformation_artifact (DataTransformationArtifact): Contains paths to transformed train/test data.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
   
    def train_model(self, X_train, y_train, x_test, y_test):
        """
        Trains and evaluates multiple models using GridSearchCV for hyperparameter tuning.
        Logs the best model and metrics to MLflow.
        
        Parameters:
            X_train, y_train: Training data and labels.
            x_test, y_test: Testing data and labels.
        
        Returns:
            ModelTrainerArtifact: Contains information about the trained model and its performance.
        """
        try:
            # Define models and their hyperparameters
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models and find the best one
            model_report: dict = evaluate_models(X_train, y_train, x_test, y_test, models=models, param=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
           
            # Log metrics for train and test datasets and track with mlflow
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            self.track_mlflow(best_model, classification_train_metric)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model, classification_test_metric)

            # Save the model and preprocessor
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
            save_object("final_model/model.pkl", best_model)

            # Return artifact with training details
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        """
        Logs metrics and the model to MLflow for experiment tracking.

        Parameters:
            best_model: The trained model.
            classificationmetric: Metrics (F1, precision, recall) for the model.
        """
        try:
            # Set MLflow tracking URI
            #mlflow.set_registry_uri("https://dagshub.com/krishnaik06/networksecurity.mlflow")
            #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run():
                # Log metrics to MLflow
                mlflow.log_metric("f1_score", classificationmetric.f1_score)
                mlflow.log_metric("precision", classificationmetric.precision_score)
                mlflow.log_metric("recall", classificationmetric.recall_score)

                # Log the trained model
                mlflow.sklearn.log_model(best_model, "model")
                
                #if tracking_url_type_store != "file":
                    #mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Loads transformed data, trains models, and returns the ModelTrainerArtifact.
        
        Returns:
            ModelTrainerArtifact: Contains details of the trained model and its performance.
        """
        try:
            # Load train and test data arrays
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],  # Features
                train_arr[:, -1],   # Labels
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Train models and return the artifact
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)