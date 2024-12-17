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