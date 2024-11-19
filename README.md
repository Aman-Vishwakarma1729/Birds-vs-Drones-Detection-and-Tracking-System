# Birds vs Drones Detection and Tracking System

## 🚀 Project Overview
A comprehensive computer vision application for detecting and tracking birds and drones using YOLOv8. This system employs advanced deep learning techniques with a focus on modularity, error handling, and robust logging capabilities.

## 🛠 Key Features
- Real-time bird and drone detection
- Comprehensive data augmentation pipeline
- Advanced model evaluation metrics
- Object tracking capabilities
- Robust error handling and logging
- Streamlit-based user interface
- Support for both image and video processing

## 📂 Project Structure
```
birds_vs_drones_detection_and_tracking/
├── .env                        # Environment variables configuration
├── .gitignore                 # Git ignore rules
├── app.py                     # Streamlit web application
├── requirements.txt           # Project dependencies
├── setup.py                  # Package installation configuration
├── yolov8n.pt                # YOLOv8 nano pre-trained weights
├── augmented_data/           # Directory for augmented training data
├── data/                     # Dataset directory
│   ├── README.dataset.txt    # Dataset documentation
│   ├── README.roboflow.txt   # Roboflow dataset information
│   ├── data.yaml            # Dataset configuration
│   ├── train/               # Training dataset
│   ├── valid/               # Validation dataset
│   └── test/                # Test dataset
├── research/                 # Research and development notebooks
│   ├── edith-defence-system-v-0.0.1.ipynb
│   └── edith-defence-system-v-0.0.2.ipynb
├── runs/                     # Training runs and model artifacts
└── src/                     # Source code
    ├── __init__.py
    ├── components/          # Core components
    │   ├── __init__.py
    │   ├── data_augmentation.py    # Data augmentation pipeline
    │   └── download_dataset.py     # Dataset download utilities
    ├── pipeline/            # Training and inference pipelines
    │   ├── __init__.py
    │   ├── evaluation.py           # Model evaluation scripts
    │   ├── prediction.py           # Prediction pipeline
    │   └── training-v-0.0.1.py     # Training pipeline
    ├── custom_exception.py  # Custom exception handling
    └── logger.py           # Logging infrastructure
```

### 📁 Directory Details

#### Core Directories
- **src/**: Contains all source code and core functionality
  - **components/**: Core processing modules
  - **pipeline/**: Training and inference pipelines
  - **custom_exception.py**: Exception handling system
  - **logger.py**: Logging infrastructure

#### Data Management
- **data/**: Contains dataset files and configuration
  - Organized into train, validation, and test sets
  - Includes dataset documentation and configuration
- **augmented_data/**: Stores augmented training data
  - Generated by data_augmentation.py
  - Used for model training enhancement

#### Model Artifacts
- **runs/**: Contains training outputs
  - Model weights
  - Training logs
  - Evaluation metrics
- **yolov8n.pt**: Pre-trained YOLOv8 nano weights

#### Development
- **research/**: Jupyter notebooks for R&D
  - Version 0.0.1 and 0.0.2 development notebooks
  - Experimental features and analysis

#### Configuration
- **.env**: Environment variables
- **requirements.txt**: Python dependencies
- **.gitignore**: Git ignore patterns

## 🔧 Component Details

### Core Components
1. **Data Augmentation** (`components/data_augmentation.py`)
   - Implements advanced augmentation techniques
   - Supports various transformation strategies
   - Handles batch processing of images

2. **Dataset Management** (`components/download_dataset.py`)
   - Manages dataset download from Roboflow
   - Handles data validation and verification
   - Implements error handling for download process

3. **Model Pipeline**
   - **Evaluation** (`pipeline/evaluation.py`): Comprehensive model evaluation metrics
   - **Prediction** (`pipeline/prediction.py`): Real-time inference pipeline
   - **Training** (`pipeline/training-v-0.0.1.py`): YOLOv8 training workflow

### Infrastructure
1. **Exception Handling** (`custom_exception.py`)
   - Custom exception classes for different error types
   - Detailed error tracking with file and line information
   - Specialized exceptions for model, data, and prediction errors

2. **Logging System** (`logger.py`)
   - Comprehensive logging infrastructure
   - Multiple logger categories (model, data, prediction)
   - Timestamped log files and console output

## 📦 Dependencies
- ultralytics==8.0.0
- torch>=2.0.0
- opencv-python>=4.7.0
- streamlit>=1.24.0
- albumentations>=1.3.1
- python-dotenv>=1.0.0
- numpy>=1.24.0
- matplotlib>=3.7.1
- pandas>=2.0.0

## 🚀 Getting Started

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Installation**
   There are two ways to install the project dependencies:

   a) Using requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

   b) Using setup.py (recommended for development):
   ```bash
   pip install -e .
   ```
   This will install the project in development mode, making it easier to modify the code without reinstalling.

   The setup.py file includes the following dependencies:
   - ultralytics (YOLOv8)
   - numpy
   - matplotlib
   - pandas
   - opencv-python
   - streamlit
   - albumentations
   - python-dotenv

3. **Configure Environment Variables**
   - Create a `.env` file in the project root
   - Add your Roboflow API key:
     ```
     ROBOFLOW_API_KEY=your_api_key_here
     ```

4. **Download Dataset**
   ```bash
   python src/components/download_dataset.py
   ```

5. **Run Training**
   ```bash
   python src/pipeline/training-v-0.0.1.py
   ```

6. **Launch Web Interface**
   ```bash
   streamlit run app.py
   ```

## 🔍 Model Configuration
- Base Model: YOLOv8 Nano
- Classes: Bird, Drone
- Default Confidence Threshold: 0.25
- IoU Threshold Range: 0.45-0.5

## 📊 Evaluation Metrics
- mAP50
- mAP50-95
- Precision
- Recall
- F1-score

## 🔐 Security Considerations
- API keys stored in `.env` file
- Secure file handling implementation
- Input validation for all user inputs
- Comprehensive error logging

## 🚧 Future Improvements
1. Enhanced small object detection capabilities
2. Implementation of advanced tracking algorithms
3. Real-time camera input support
4. Extended dataset diversity
5. Edge device deployment optimizations
6. Comprehensive unit test coverage

## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
