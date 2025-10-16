# python logger.py


import logging
import os
from datetime import datetime

import sys


# Define the desired logs directory
logs_dir = "/Users/apple/Downloads/Data_science_file/Agent8/Projects/Treking_cost_predictor/src/logs"

today_date = datetime.now().strftime("%d_%m_%Y")
daily_log_dir = os.path.join(logs_dir, today_date)

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) # use if you want to in current working directory
os.makedirs(daily_log_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(daily_log_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",  # Custom date format with milliseconds
    level=logging.INFO,
)


"""class DataIngestion:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Exited the data ingestion method or component")
            return None
        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e, sys)"""


if __name__ == "__main__":
    logging.info("Logging has started")
    """obj = DataIngestion()
    obj.initiate_data_ingestion()"""
