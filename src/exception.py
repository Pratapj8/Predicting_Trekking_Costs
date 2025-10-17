# exception.py

import sys  # Importing sys module to get error details
import logging
import os

from src.logger import logging


# Function to get detailed error message
def error_messege_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Get traceback object

    if exc_tb is not None:
        file_name = (
            exc_tb.tb_frame.f_code.co_filename
        )  # Get file name where error happened
        error_line_no = exc_tb.tb_lineno  # Get line number where error happened
    else:
        file_name = "Unknown"
        error_line_no = "Unknown"

    # Format error message with file, line, and error
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, error_line_no, str(error)
    )
    return error_message  # Return the full error message


# Custom exception class to show error in our own way
class CustomException(Exception):
    def __init__(
        self, error_message, error_detail: sys
    ):  # Constructor takes error and details
        super().__init__(error_message)  # Call base class constructor
        self.error_message = error_messege_detail(
            error_message, error_detail
        )  # Get and store detailed error

    def __str__(self):  # When we print the exception
        return self.error_message  # Show the detailed error message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e, sys)
