from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
import time
from src.logger import prediction_logger
from src.custom_exception import PredictionException
import sys

class ObjectDetector:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        """
        Initialize the object detector
        
        Args:
            model_path (str): Path to the trained model weights
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
        """
        try:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise PredictionException(f"Model path does not exist: {model_path}", sys)
            
            prediction_logger.info(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            
            # Class mapping
            self.class_names = ['bird', 'drone']
            
        except Exception as e:
            prediction_logger.error(f"Error initializing ObjectDetector: {str(e)}")
            raise PredictionException(e, sys)
    
    def predict_image(self, image_path, save_path=None):
        """
        Run prediction on a single image
        
        Args:
            image_path (str): Path to input image
            save_path (str, optional): Path to save annotated image
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise PredictionException(f"Image path does not exist: {image_path}", sys)
            
            prediction_logger.info(f"Running prediction on {image_path}")
            
            # Run prediction
            results = self.model(str(image_path), conf=self.conf_thres, iou=self.iou_thres)[0]
            
            # Process results
            predictions = []
            for box in results.boxes:
                pred = {
                    'class': self.class_names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                predictions.append(pred)
            
            # Save annotated image if requested
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                annotated_img = results.plot()
                cv2.imwrite(str(save_path), annotated_img)
            
            prediction_logger.info(f"Found {len(predictions)} objects")
            return predictions
            
        except Exception as e:
            prediction_logger.error(f"Error in image prediction: {str(e)}")
            raise PredictionException(e, sys)
    
    def predict_video(self, video_path, save_path=None, track=True):
        """
        Run prediction on video
        
        Args:
            video_path (str): Path to input video
            save_path (str, optional): Path to save annotated video
            track (bool): Whether to use object tracking
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                raise PredictionException(f"Video path does not exist: {video_path}", sys)
            
            prediction_logger.info(f"Running prediction on video {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise PredictionException(f"Error opening video file: {video_path}", sys)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer if save_path is provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
            
            # Process video frames
            frame_count = 0
            results_list = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run prediction
                if track:
                    results = self.model.track(frame, conf=self.conf_thres, iou=self.iou_thres, persist=True)[0]
                else:
                    results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres)[0]
                
                # Process results
                frame_predictions = []
                for box in results.boxes:
                    pred = {
                        'frame': frame_count,
                        'class': self.class_names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()
                    }
                    if track and hasattr(box, 'id'):
                        pred['track_id'] = int(box.id)
                    frame_predictions.append(pred)
                
                results_list.append(frame_predictions)
                
                # Save frame if requested
                if save_path:
                    annotated_frame = results.plot()
                    out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    prediction_logger.info(f"Processed {frame_count}/{total_frames} frames")
            
            # Clean up
            cap.release()
            if save_path:
                out.release()
            
            prediction_logger.info(f"Video processing completed. Processed {frame_count} frames")
            return results_list
            
        except Exception as e:
            prediction_logger.error(f"Error in video prediction: {str(e)}")
            raise PredictionException(e, sys)
    
    def export_results(self, results, output_path):
        """
        Export prediction results to JSON
        
        Args:
            results (list): List of predictions
            output_path (str): Path to save JSON file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            prediction_logger.info(f"Exporting results to {output_path}")
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            prediction_logger.info("Results exported successfully")
            
        except Exception as e:
            prediction_logger.error(f"Error exporting results: {str(e)}")
            raise PredictionException(e, sys)

def main():
    try:
        # Define paths
        model_path = "runs/train/yolov8n_birds_drones/weights/best.pt"
        image_path = "data/test/images/test_image.jpg"
        video_path = "data/test/videos/test_video.mp4"
        
        # Create detector
        detector = ObjectDetector(model_path)
        
        # Test image prediction
        image_results = detector.predict_image(
            image_path,
            save_path="predictions/test_image_pred.jpg"
        )
        detector.export_results(image_results, "predictions/image_results.json")
        
        # Test video prediction
        video_results = detector.predict_video(
            video_path,
            save_path="predictions/test_video_pred.mp4",
            track=True
        )
        detector.export_results(video_results, "predictions/video_results.json")
        
        prediction_logger.info("Prediction pipeline test completed successfully!")
        
    except Exception as e:
        prediction_logger.error(f"Error in main prediction process: {str(e)}")
        raise PredictionException(e, sys)

if __name__ == "__main__":
    main()
