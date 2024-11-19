from roboflow import Roboflow
import os
import shutil
import yaml
from pathlib import Path
from dotenv import load_dotenv
from src.logger import data_logger
from src.custom_exception import DataException
import sys

def update_data_yaml(project_root: Path, downloaded_path: Path):
    """
    Update data.yaml with correct paths after dataset download
    
    Args:
        project_root (Path): Path to project root directory
        downloaded_path (Path): Path to downloaded dataset directory
    """
    try:
        # Read existing data.yaml
        yaml_path = downloaded_path / "data.yaml"
        if not yaml_path.exists():
            raise DataException(f"data.yaml not found in {downloaded_path}", sys)
            
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths to be relative to project root
        data_config['path'] = str(project_root / 'data')
        data_config['train'] = './train/images'
        data_config['val'] = './valid/images'
        data_config['test'] = './test/images'
        
        # Write updated data.yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
            
        data_logger.info("Updated data.yaml with correct paths")
        
    except Exception as e:
        data_logger.error(f"Error updating data.yaml: {str(e)}")
        raise DataException(e, sys)

def rename_dataset_folder(downloaded_path: Path, target_path: Path):
    """
    Rename downloaded dataset folder to 'data'
    
    Args:
        downloaded_path (Path): Path to downloaded dataset directory
        target_path (Path): Target path for dataset directory
    """
    try:
        if target_path.exists():
            data_logger.warning(f"Target directory {target_path} already exists. Removing...")
            shutil.rmtree(target_path)
        
        data_logger.info(f"Moving dataset from {downloaded_path} to {target_path}")
        shutil.move(str(downloaded_path), str(target_path))
        data_logger.info("Dataset folder renamed successfully")
        
    except Exception as e:
        data_logger.error(f"Error renaming dataset folder: {str(e)}")
        raise DataException(e, sys)

def download_dataset():
    """
    Download and setup dataset from Roboflow
    
    Returns:
        Path: Path to the dataset directory
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        
        # Load environment variables
        data_logger.info("Loading environment variables...")
        load_dotenv()
        api_key = os.getenv('ROBOFLOW_API_KEY')
        
        if not api_key:
            raise DataException("Roboflow API key not found in environment variables", sys)
        
        data_logger.info("Initializing Roboflow client...")
        rf = Roboflow(api_key=api_key)
        
        data_logger.info("Accessing project workspace...")
        project = rf.workspace("cuiwah").project("uav-wjqyf")
        version = project.version(1)
        
        data_logger.info("Downloading dataset...")
        dataset = version.download("yolov8")
        
        # Get downloaded dataset path (usually named after the project)
        downloaded_path = project_root / "uav-wjqyf-1"
        if not downloaded_path.exists():
            raise DataException(f"Downloaded dataset not found at {downloaded_path}", sys)
        
        # Rename dataset folder to 'data'
        target_path = project_root / "data"
        rename_dataset_folder(downloaded_path, target_path)
        
        # Update data.yaml with correct paths
        update_data_yaml(project_root, target_path)
        
        data_logger.info("Dataset setup completed successfully!")
        return target_path
        
    except Exception as e:
        data_logger.error(f"Error in dataset download and setup: {str(e)}")
        raise DataException(e, sys)

def main():
    try:
        download_dataset()
    except Exception as e:
        data_logger.error(f"Dataset download failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()