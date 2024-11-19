import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path
from src.logger import prediction_logger
from src.custom_exception import PredictionException
import sys

class StreamlitApp:
    def __init__(self, model_path):
        """
        Initialize the Streamlit application
        
        Args:
            model_path (str): Path to the trained model weights
        """
        try:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise PredictionException(f"Model path does not exist: {model_path}", sys)
            
            prediction_logger.info(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
            
        except Exception as e:
            prediction_logger.error(f"Error initializing StreamlitApp: {str(e)}")
            raise PredictionException(e, sys)
    
    def process_image(self, image):
        """
        Process a single image through the YOLOv8 detection model
        
        Args:
            image: PIL Image or numpy array
        """
        try:
            # Convert PIL Image to numpy array if necessary
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            prediction_logger.info("Running inference on image")
            results = self.model(image_np)
            
            # Get the first result (assuming single image)
            result = results[0]
            
            # Create a copy of the image for drawing
            annotated_image = image_np.copy()
            
            # List to store predictions
            predictions = []
            
            # Process each detection
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Store prediction
                predictions.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
            
            prediction_logger.info(f"Found {len(predictions)} objects in image")
            return annotated_image, predictions
            
        except Exception as e:
            prediction_logger.error(f"Error processing image: {str(e)}")
            raise PredictionException(e, sys)
    
    def process_video_frame(self, frame):
        """
        Process a video frame and return the annotated frame
        
        Args:
            frame: Video frame as numpy array
        """
        try:
            processed_frame, _ = self.process_image(frame)
            return processed_frame
            
        except Exception as e:
            prediction_logger.error(f"Error processing video frame: {str(e)}")
            raise PredictionException(e, sys)
    
    def run(self):
        """Run the Streamlit application"""
        try:
            st.title("Birds vs Drones Detection and Tracking")
            prediction_logger.info("Starting Streamlit application")
            
            # Create tabs for image and video processing
            tab1, tab2 = st.tabs(["Image Detection", "Video Detection & Tracking"])
            
            with tab1:
                st.header("Upload Image for Detection")
                image_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'], key="image_uploader")
                
                if image_file is not None:
                    try:
                        prediction_logger.info(f"Processing uploaded image: {image_file.name}")
                        
                        # Read and process image
                        image = Image.open(image_file)
                        
                        # Create columns for original and processed images
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Original Image")
                            st.image(image)
                        
                        with col2:
                            st.subheader("Detected Objects")
                            processed_image, predictions = self.process_image(image)
                            st.image(processed_image)
                        
                        # Display detection results
                        st.subheader("Detection Results")
                        if predictions:
                            for pred in predictions:
                                st.write(f"Detected {pred['class']} with {pred['confidence']:.2f} confidence")
                        else:
                            st.write("No objects detected")
                            
                    except Exception as e:
                        prediction_logger.error(f"Error processing image {image_file.name}: {str(e)}")
                        st.error(f"Error processing image: {str(e)}")
            
            with tab2:
                st.header("Upload Video for Detection & Tracking")
                video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'], key="video_uploader")
                
                if video_file is not None:
                    try:
                        prediction_logger.info(f"Processing uploaded video: {video_file.name}")
                        
                        # Save uploaded video to temporary file
                        tfile = tempfile.NamedTemporaryFile(delete=False)
                        tfile.write(video_file.read())
                        
                        # Open video file
                        cap = cv2.VideoCapture(tfile.name)
                        
                        if not cap.isOpened():
                            raise PredictionException(f"Error opening video file: {video_file.name}", sys)
                        
                        # Get video properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        prediction_logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
                        
                        # Create a placeholder for the video
                        video_placeholder = st.empty()
                        
                        # Process and display video frames
                        frame_count = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Process frame
                            processed_frame = self.process_video_frame(frame)
                            
                            # Convert BGR to RGB
                            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display the processed frame
                            video_placeholder.image(processed_frame_rgb)
                            
                            frame_count += 1
                            if frame_count % 100 == 0:
                                prediction_logger.info(f"Processed {frame_count}/{total_frames} frames")
                        
                        cap.release()
                        prediction_logger.info(f"Completed video processing: {frame_count} frames processed")
                        
                    except Exception as e:
                        prediction_logger.error(f"Error processing video {video_file.name}: {str(e)}")
                        st.error(f"Error processing video: {str(e)}")
            
        except Exception as e:
            prediction_logger.error(f"Error in Streamlit application: {str(e)}")
            st.error("An unexpected error occurred. Please check the logs for details.")

def main():
    try:
        # Initialize and run the application
        MODEL_PATH = "D:/External_Projects/Edith_Defene_System/birds_vs_drones_detection_and_tracking_version-0.0.1/runs/train/yolov8n_birds_drones/weights/best.pt"
        app = StreamlitApp(MODEL_PATH)
        app.run()
        
    except Exception as e:
        prediction_logger.error(f"Error in main: {str(e)}")
        st.error("Failed to initialize the application. Please check the logs for details.")

if __name__ == "__main__":
    main()
