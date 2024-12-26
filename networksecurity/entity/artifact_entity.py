# Import the dataclass decorator to simplify class creation for data storage
from dataclasses import dataclass 

# Define a data class to store file paths generated during data ingestion
@dataclass
class DataIngestionArtifact:
    """
    DataIngestionArtifact is a data class used to store:
        - trained_file_path: Path to the training dataset file.
        - test_file_path: Path to the testing dataset file.
    """
    trained_file_path: str
    test_file_path: str

@dataclass    
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact