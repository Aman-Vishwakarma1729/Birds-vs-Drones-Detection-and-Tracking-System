from roboflow import Roboflow
import os
from dotenv import load_dotenv
from src.logger import data_logger
from src.custom_exception import DataException
import sys

def download_dataset():
    """Download dataset from Roboflow"""
    try:
        # Load environment variables
        data_logger.info("Loading environment variables...")
        load_dotenv()
        api_key = os.getenv('robolflow_api_key')
        
        if not api_key:
            raise DataException("Roboflow API key not found in environment variables", sys)
        
        data_logger.info("Initializing Roboflow client...")
        rf = Roboflow(api_key=api_key)
        
        data_logger.info("Accessing project workspace...")
        project = rf.workspace("cuiwah").project("uav-wjqyf")
        version = project.version(1)
        
        data_logger.info("Downloading dataset...")
        dataset = version.download("yolov8")
        
        data_logger.info("Dataset downloaded successfully!")
        return dataset
        
    except Exception as e:
        data_logger.error(f"Error in dataset download: {str(e)}")
        raise DataException(e, sys)

if __name__ == "__main__":
    download_dataset()