import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

# @dataclass helps to create classes which mainly store data with less boilerplate code.
# it automatically generates special methods like __init__() and __repr__() based on class attributes.
@dataclass 
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
# self  -----------------------> current DataIngestion object
#     │
#     └── ingestion_config ----> DataIngestionConfig object
#            │
#            ├── raw_data_path
#            ├── train_data_path
#            └── test_data_path

    def initiate_data_ingestion(self,source_file_path: str):
        logging.info("Starting data ingestion process")
        try:
            # Read the dataset
            df = pd.read_csv(source_file_path)
            logging.info("Dataset read successfully")

            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False , header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Dataset split into training and testing sets")

            # Save the training data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            logging.info(f"Training data saved at {self.ingestion_config.train_data_path}")

            # Save the testing data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info(f"Testing data saved at {self.ingestion_config.test_data_path}")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys) from e
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion(source_file_path="notebook/data/stud.csv")

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    ModelTrainer = ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_arr,test_arr))