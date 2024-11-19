import logging
import os
from datetime import datetime

# Create logs directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add console handler to root logger
logging.getLogger('').addHandler(console_handler)

# Create logger categories
model_logger = logging.getLogger('model')
data_logger = logging.getLogger('data')
prediction_logger = logging.getLogger('prediction')
evaluation_logger = logging.getLogger('evaluation')

# Example usage
if __name__ == "__main__":
    logging.info("Logging test message")
    model_logger.info("Model-specific log message")
    data_logger.warning("Data-related warning message")
    prediction_logger.error("Prediction error message")
    evaluation_logger.info("Evaluation status message")
