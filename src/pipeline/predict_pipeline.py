import os
import sys
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features_df):
        try:
            logging.info("Loading preprocessor and model...")

            # Load artifacts
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            logging.info("Transforming features...")
            data_scaled = preprocessor.transform(features_df)

            logging.info("Making predictions...")
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Converts user input into a DataFrame format
    that matches the training data.
    """

    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            data = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # Alias for compatibility with app.py
    def get_data_as_data_frame(self):
        return self.get_data_as_dataframe()
