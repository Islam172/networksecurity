import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constants.training_pipeline import TARGET_COLUMN
from networksecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    """
    This class handles the data transformation process, including:
    1. Imputation of missing values.
    2. Preparation of transformed data for machine learning models.
    3. Saving the transformed data and preprocessing object.
    """
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        """
        Initializes the DataTransformation class with necessary artifacts and configurations.

        Parameters:
            data_validation_artifact (DataValidationArtifact): Contains paths to valid train and test data.
            data_transformation_config (DataTransformationConfig): Configuration for data transformation.
        """
        try:
            # Initialize class attributes
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            # Handle initialization errors with a custom exception
            raise NetworkSecurityException(e, sys)
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads data from a CSV file and returns it as a Pandas DataFrame.

        Parameters:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        try:
            return pd.read_csv(file_path)  # Read the CSV file into a DataFrame
        except Exception as e:
            # Raise a custom exception if file reading fails
            raise NetworkSecurityException(e, sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Initializes and returns a scikit-learn Pipeline for data preprocessing.

        The pipeline includes:
        1. KNNImputer for imputing missing values.

        Returns:
            Pipeline: A preprocessing pipeline object.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            # Initialize a KNNImputer with specified parameters
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialized KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            # Create a pipeline with the KNNImputer
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            # Handle pipeline initialization errors
            raise NetworkSecurityException(e, sys)
        

       
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Orchestrates the data transformation process, including:
        1. Loading valid train and test data.
        2. Imputing missing values and transforming the datasets.
        3. Saving transformed data and the preprocessing object.

        Returns:
            DataTransformationArtifact: Contains paths to transformed data and the preprocessing object.
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            # Step 1: Read train and test datasets
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Step 2: Prepare input and target features for training data
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)  # Drop target column
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)  # Replace -1 with 0 in the target

            # Step 3: Prepare input and target features for testing data
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)  # Drop target column
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)  # Replace -1 with 0 in the target

            # Step 4: Initialize the preprocessing pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit the preprocessor on the training data and transform both datasets
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Step 5: Combine transformed input features and target features into arrays
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Step 6: Save the transformed data and preprocessing object
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # Step 7: Prepare and return the DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            # Handle transformation errors with a custom exception
            raise NetworkSecurityException(e, sys)
     
     
     
