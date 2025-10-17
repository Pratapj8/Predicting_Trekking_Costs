'''# python logger.py


import logging
import os
from datetime import datetime

import sys


# Define the desired logs directory
logs_dir = "/Users/apple/Downloads/Data_science_file/Agent8/Projects/Treking_cost_predictor/logs"


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
    obj = DataIngestion()
    obj.initiate_data_ingestion()
'''

"""
"logger" is a tool for recording events that happen while your program is running.
Think of it as a sophisticated way to keep a diary of your application's activities,
much more powerful than just using print() statements.
"""

"""
DEBUG: Detailed information, typically for debugging.
INFO: Confirmation that things are working as expected.
WARNING: An indication that something unexpected happened, or indicative of a problem in the near future (e.g., 'disk space low').
ERROR: Due to a more serious problem, the software has not been able to perform some function.
CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

"""


import logging
import os
from datetime import datetime

# Log file name

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Log directory
# cwd - current working directory
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

os.makedirs(logs_path, exist_ok=True)

# Log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


if __name__ == "__main__":
    logging.info("Logging has started")
