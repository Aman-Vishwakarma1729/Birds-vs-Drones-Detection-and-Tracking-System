from setuptools import setup, find_packages
from pathlib import Path
import os

# Read requirements from requirements.txt
def read_requirements(filename: str):
    try:
        with open(filename, 'r') as file:
            # Read non-empty lines that don't start with '#'
            return [line.strip() for line in file if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Please make sure requirements.txt exists.")
        return []

# Get project root directory and requirements path
project_root = Path(__file__).parent
req_path = project_root / 'requirements.txt'

# Get requirements
requirements = read_requirements(str(req_path))

setup(
    name="Bird vs Drone Detection and Tracking System",
    version="0.0.1",
    author="Aman Vishwakarma",
    author_email="amansharma1729ds@gmail.com",
    description="A computer vision system for detecting and tracking birds and drones using YOLOv8 nano, a lightweight object detection model",
    long_description=Path(project_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
    url="https://github.com/Aman-Vishwakarma1729/Birds-vs-Drones-Detection-and-Tracking-System",  
    keywords=["computer vision", "object detection", "YOLOv8 nano", "drone detection", "bird detection"],
)
