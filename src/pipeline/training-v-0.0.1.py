from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
from src.logger import model_logger
from src.custom_exception import ModelException
import sys

class ModelTrainer:
    def __init__(self, data_yaml_path, model_type='yolov8n.pt'):
        """
        Initialize the model trainer
        
        Args:
            data_yaml_path (str): Path to data.yaml file
            model_type (str): Type of YOLO model to use
        """
        try:
            self.data_yaml_path = Path(data_yaml_path)
            if not self.data_yaml_path.exists():
                raise ModelException(f"Data YAML file not found: {data_yaml_path}", sys)
            
            # Validate data.yaml
            with open(self.data_yaml_path) as f:
                self.data_config = yaml.safe_load(f)
                required_keys = ['path', 'train', 'val', 'test', 'nc', 'names']
                if not all(key in self.data_config for key in required_keys):
                    raise ModelException(f"Invalid data.yaml file. Required keys: {required_keys}", sys)
            
            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_logger.info(f"Using device: {self.device}")
            
            # Load model
            model_logger.info(f"Loading {model_type} model...")
            self.model = YOLO(model_type)
            
        except Exception as e:
            model_logger.error(f"Error initializing ModelTrainer: {str(e)}")
            raise ModelException(e, sys)
    
    def train(self, epochs=100, imgsz=640, batch=16, workers=4, patience=20):
        """
        Train the model
        
        Args:
            epochs (int): Number of training epochs
            imgsz (int): Input image size
            batch (int): Batch size
            workers (int): Number of worker threads
            patience (int): Early stopping patience
        """
        try:
            model_logger.info("Starting model training...")
            model_logger.info(f"Training parameters: epochs={epochs}, imgsz={imgsz}, batch={batch}, workers={workers}")
            
            # Train model
            results = self.model.train(
                data=str(self.data_yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                workers=workers,
                patience=patience,
                save=True,
                project='runs/train',
                name='yolov8n_birds_drones',
                exist_ok=True
            )
            
            # Log training metrics
            model_logger.info("Training completed. Final metrics:")
            model_logger.info(f"Box mAP: {results.box.map}")
            model_logger.info(f"Box mAP50: {results.box.map50}")
            model_logger.info(f"Box mAP75: {results.box.map75}")
            
            return results
            
        except Exception as e:
            model_logger.error(f"Error during model training: {str(e)}")
            raise ModelException(e, sys)
    
    def validate(self):
        """
        Validate the trained model
        """
        try:
            model_logger.info("Starting model validation...")
            
            # Validate model
            results = self.model.val()
            
            # Log validation metrics
            model_logger.info("Validation completed. Metrics:")
            model_logger.info(f"Box mAP: {results.box.map}")
            model_logger.info(f"Box mAP50: {results.box.map50}")
            model_logger.info(f"Box mAP75: {results.box.map75}")
            
            return results
            
        except Exception as e:
            model_logger.error(f"Error during model validation: {str(e)}")
            raise ModelException(e, sys)

def main():
    try:
        # Initialize trainer
        trainer = ModelTrainer('data/data.yaml')
        
        # Train model
        train_results = trainer.train()
        
        # Validate model
        val_results = trainer.validate()
        
        model_logger.info("Training and validation completed successfully!")
        
    except Exception as e:
        model_logger.error(f"Error in training pipeline: {str(e)}")
        raise ModelException(e, sys)

if __name__ == "__main__":
    main()
