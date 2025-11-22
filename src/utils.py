import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score

def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error("Error saving object")
        raise CustomException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            model_name = list(models.keys())[i]
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"{model_name} R2 Score: {test_model_score}")

        return report

    except Exception as e:
        logging.error("Error evaluating models")
        raise CustomException(e, sys) from e