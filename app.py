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
import base64

class StreamlitApp:
    def __init__(self, model_path):
        """Initialize the Streamlit application"""
        try:
            # Get the absolute path to the project root directory
            project_root = Path(__file__).parent
            self.model_path = project_root / model_path
            
            if not self.model_path.exists():
                raise PredictionException(f"Model path does not exist: {self.model_path}", sys)
            
            prediction_logger.info(f"Loading model from {self.model_path}")
            self.model = YOLO(str(self.model_path))
            
        except Exception as e:
            prediction_logger.error(f"Error initializing StreamlitApp: {str(e)}")
            raise PredictionException(e, sys)
    
    @staticmethod
    def set_page_config():
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="Birds vs Drones Detection and Tracking System",
            page_icon="🦅",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    @staticmethod
    def add_bg_from_local(image_file):
        """Add background image to the app"""
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    def render_header(self):
        """Render the app header"""
        st.markdown(
            """
            <div style='text-align: center; background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 10px;'>
                <h1 style='color: #00ff00;'>🦅 Bird vs Drone Detection and Tracking System 🛸</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_sidebar(self):
        """Render the sidebar with navigation"""
        with st.sidebar:
            st.markdown(
                """
                <div style='background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 10px;'>
                    <h2 style='color: #00ff00;'>Navigation</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            pages = {
                "🏠 Home": "home",
                "📊 Detection System": "detection",
                "ℹ️ About": "about"
            }
            
            selected_page = st.radio("", list(pages.keys()))
            return pages[selected_page]
    
    def render_home_page(self):
        """Render the home page"""
        st.markdown(
            """
            <div style='background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 10px;'>
                <h2 style='color: #00ff00;'>Welcome to real time Birds vs Drones Detection and Tracking System</h2>
                <p style='color: #ffffff;'>
                    A state-of-the-art computer vision system for detecting and tracking birds and drones using YOLOv8 nano,
                    a lightweight object detection model.
                </p>
                <h3 style='color: #00ff00;'>Key Features:</h3>
                <ul style='color: #ffffff;'>
                    <li>Real-time bird and drone detection</li>
                    <li>Advanced tracking capabilities</li>
                    <li>Support for both image and video processing</li>
                    <li>Comprehensive logging system</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Add GitHub link
        st.markdown(
            """
            <div style='background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 10px; margin-top: 20px;'>
                <h3 style='color: #00ff00;'>Resources:</h3>
                <ul style='color: #ffffff;'>
                    <li>🔗 <a href='https://github.com/Aman-Vishwakarma1729/Birds-vs-Drones-Detection-and-Tracking-System' style='color: #00ff00;'>GitHub Repository</a></li>
                    <li>📊 <a href='' style='color: #00ff00;'>Trial Dataset</a></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_about_page(self):
        """Render the about page"""
        st.markdown(
            """
            <div style='background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 10px;'>
                <h2 style='color: #00ff00;'>About the Project</h2>
                <p style='color: #ffffff;'>
                    The EDITH Defense System is designed to enhance security and surveillance capabilities by accurately 
                    detecting and tracking both birds and drones. Using the lightweight YOLOv8 nano model, it provides 
                    efficient and real-time object detection while maintaining high accuracy.
                </p>
                <h3 style='color: #00ff00;'>Technical Details:</h3>
                <ul style='color: #ffffff;'>
                    <li>Model: YOLOv8 nano</li>
                    <li>Framework: Ultralytics</li>
                    <li>Interface: Streamlit</li>
                    <li>Computer Vision: OpenCV</li>
                </ul>
                <h3 style='color: #00ff00;'>Developer:</h3>
                <p style='color: #ffffff;'>
                    Developed by Aman Vishwakarma<br>
                    <a href='mailto:amansharma1729ds@gmail.com' style='color: #00ff00;'>amansharma1729ds@gmail.com</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_detection_page(self):
        """Render the detection system page"""
        st.markdown(
            """
            <div style='background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='color: #00ff00;'>Detection System</h2>
                <p style='color: #ffffff;'>Upload an image or video to detect birds and drones.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
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
        self.set_page_config()
        # Add background image (you'll need to add an appropriate image)
        # self.add_bg_from_local("path_to_your_background_image.png")
        
        self.render_header()
        page = self.render_sidebar()
        
        if page == "home":
            self.render_home_page()
        elif page == "detection":
            self.render_detection_page()
        elif page == "about":
            self.render_about_page()

def main():
    model_path = "runs/train/yolov8n_birds_drones/weights/best.pt"
    app = StreamlitApp(model_path)
    app.run()

if __name__ == "__main__":
    main()
