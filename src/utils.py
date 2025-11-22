import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
import dill

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