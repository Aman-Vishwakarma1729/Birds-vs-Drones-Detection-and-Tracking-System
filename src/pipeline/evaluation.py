from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import sys

# Add the project root directory to the Python path when running directly
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.logger import evaluation_logger
    from src.custom_exception import ValidationException
else:
    from ..logger import evaluation_logger
    from ..custom_exception import ValidationException

class ModelEvaluator:
    def __init__(self, model_path, data_yaml_path):
        """
        Initialize the model evaluator
        
        Args:
            model_path (str): Path to the trained model weights
            data_yaml_path (str): Path to the data.yaml file
        """
        try:
            self.model_path = Path(model_path)
            self.data_yaml_path = Path(data_yaml_path)
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            if not self.data_yaml_path.exists():
                raise FileNotFoundError(f"Data YAML path does not exist: {data_yaml_path}")
            
            evaluation_logger.info(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
            
        except Exception as e:
            evaluation_logger.error(f"Error initializing ModelEvaluator: {str(e)}")
            raise ValidationException(str(e), sys) from e
    
    def evaluate_model(self, split='val'):
        """
        Evaluate the model on validation or test set
        
        Args:
            split (str): Dataset split to evaluate on ('val' or 'test')
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            if split not in ['val', 'test']:
                raise ValueError(f"Invalid split value: {split}. Must be either 'val' or 'test'")
            
            evaluation_logger.info(f"Evaluating model on {split} split")
            
            # Run validation using the model's validate method
            metrics = self.model.val(data=str(self.data_yaml_path), split=split)
            
            evaluation_logger.info(f"Evaluation metrics for {split} split: {metrics}")
            return metrics
            
        except Exception as e:
            evaluation_logger.error(f"Error during model evaluation: {str(e)}")
            raise ValidationException(str(e), sys) from e
    
    def plot_metrics(self, results, output_dir):
        """
        Plot evaluation metrics and save figures
        
        Args:
            results (dict): Dictionary containing evaluation metrics
            output_dir (str): Directory to save plots
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            evaluation_logger.info(f"Plotting metrics and saving to {output_dir}")
            
            # Bar plot of metrics
            plt.figure(figsize=(10, 6))
            plt.bar(results.keys(), results.values())
            plt.title('Model Performance Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'metrics_summary.png')
            plt.close()
            
            # Save metrics to CSV
            pd.DataFrame([results]).to_csv(output_dir / 'metrics.csv', index=False)
            
            evaluation_logger.info("Metrics plots and CSV saved successfully")
            
        except Exception as e:
            evaluation_logger.error(f"Error plotting metrics: {str(e)}")
            raise ValidationException(e, sys)
    
    def analyze_predictions(self, test_images_dir, output_dir, conf_thres=0.25):
        """
        Analyze model predictions on test images
        
        Args:
            test_images_dir (str): Directory containing test images
            output_dir (str): Directory to save analysis results
            conf_thres (float): Confidence threshold for predictions
        """
        try:
            test_images_dir = Path(test_images_dir)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            evaluation_logger.info(f"Analyzing predictions on images from {test_images_dir}")
            
            # Run predictions on test images
            results = self.model(str(test_images_dir / '*.jpg'), conf=conf_thres, save=True, save_txt=True)
            
            # Save predictions to output directory
            for result in results:
                result.save_txt(output_dir / 'labels')
                result.save_crop(output_dir / 'crops')
            
            evaluation_logger.info(f"Prediction analysis completed. Results saved to {output_dir}")
            
        except Exception as e:
            evaluation_logger.error(f"Error analyzing predictions: {str(e)}")
            raise ValidationException(e, sys)

def main():
    """
    Main function to run model evaluation
    """
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        
        # Define paths relative to project root
        model_path = r"D:\External_Projects\Edith_Defene_System\birds_vs_drones_detection_and_tracking_version-0.0.1\runs\train\yolov8n_birds_drones\weights\best.pt"
        data_yaml_path = r"D:\External_Projects\Edith_Defene_System\birds_vs_drones_detection_and_tracking_version-0.0.1\data\data.yaml"
        
        evaluator = ModelEvaluator(model_path, data_yaml_path)
        
        # Evaluate on validation set
        evaluation_logger.info("Starting validation set evaluation...")
        val_metrics = evaluator.evaluate_model(split='val')
        evaluation_logger.info("Validation Metrics:")
        evaluation_logger.info(val_metrics)
        
        # Evaluate on test set
        evaluation_logger.info("Starting test set evaluation...")
        test_metrics = evaluator.evaluate_model(split='test')
        evaluation_logger.info("Test Metrics:")
        evaluation_logger.info(test_metrics)
        
    except Exception as e:
        evaluation_logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
