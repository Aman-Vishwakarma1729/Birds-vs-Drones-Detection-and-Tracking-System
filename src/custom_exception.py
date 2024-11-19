import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Generate detailed error message with file name and line number
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    """Custom exception class for handling application specific errors"""
    
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        return self.error_message

class ModelException(CustomException):
    """Exception for model-related errors"""
    pass

class DataException(CustomException):
    """Exception for data-related errors"""
    pass

class PredictionException(CustomException):
    """Exception for prediction-related errors"""
    pass

class ValidationException(CustomException):
    """Exception for validation-related errors"""
    pass

# Example usage
if __name__ == "__main__":
    try:
        # Simulate an error
        raise ValueError("Sample error")
    except Exception as e:
        logging.error("Sample error occurred")
        raise CustomException(e, sys)
