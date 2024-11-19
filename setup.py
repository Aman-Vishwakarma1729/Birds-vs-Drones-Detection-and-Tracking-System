from setuptools import setup, find_packages

setup(
    name="edith_defense_system",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "numpy",
        "matplotlib",
        "pandas",
        "opencv-python",
        "streamlit",
        "albumentations",
        "python-dotenv",
    ],
)
