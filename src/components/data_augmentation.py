import albumentations as A
import cv2
import os
import numpy as np
from pathlib import Path
import shutil
import sys
from src.logger import data_logger
from src.custom_exception import DataException
from tqdm import tqdm

class DataAugmentation:
    def __init__(self, data_dir, output_dir):
        """
        Initialize data augmentation pipeline
        
        Args:
            data_dir (str): Path to data directory containing train, test, valid folders
            output_dir (str): Path to output directory for augmented data
        """
        try:
            self.data_dir = Path(data_dir)
            self.output_dir = Path(output_dir)
            
            # Create augmented data directories
            for split in ['train', 'test', 'valid']:
                (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
                (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            data_logger.info(f"Initialized DataAugmentation with data_dir: {data_dir}, output_dir: {output_dir}")
            
            # Define augmentation pipeline with correct transformation names
            self.transform = A.Compose([
                # Color augmentations
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                ], p=0.5),
                
                # Noise and blur for different weather conditions
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.Blur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),
                ], p=0.5),
                
                # Geometric transformations
                A.OneOf([
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-30, 30), p=0.5),
                ], p=0.5),
                
                # Weather and lighting effects
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                ], p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        except Exception as e:
            data_logger.error(f"Error initializing DataAugmentation: {str(e)}")
            raise DataException(e, sys)
    
    def process_image_and_label(self, image_path, label_path, output_split):
        """Process a single image and its label file"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                data_logger.warning(f"Warning: Could not read image {image_path}")
                return
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read YOLO format labels
            bboxes = []
            class_labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(int(class_id))
            
            # Copy original files
            shutil.copy2(image_path, self.output_dir / output_split / 'images' / image_path.name)
            if label_path.exists():
                shutil.copy2(label_path, self.output_dir / output_split / 'labels' / label_path.name)
            
            # Apply augmentations
            if bboxes:
                for i in range(3):  # Create 3 augmented versions of each image
                    try:
                        transformed = self.transform(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels
                        )
                        
                        # Save augmented image
                        aug_image_name = f"{image_path.stem}_aug_{i}{image_path.suffix}"
                        aug_label_name = f"{label_path.stem}_aug_{i}.txt"
                        
                        # Save augmented image
                        aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(
                            str(self.output_dir / output_split / 'images' / aug_image_name),
                            aug_image
                        )
                        
                        # Save augmented labels
                        with open(self.output_dir / output_split / 'labels' / aug_label_name, 'w') as f:
                            for bbox, class_label in zip(transformed['bboxes'], transformed['class_labels']):
                                f.write(f"{class_label} {' '.join(map(str, bbox))}\n")
                    except Exception as e:
                        data_logger.error(f"Warning: Failed to augment {image_path.name} (augmentation {i}): {str(e)}")
                        continue
        
        except Exception as e:
            data_logger.error(f"Error processing image and label {image_path}: {str(e)}")
            raise DataException(e, sys)
    
    def process_dataset(self):
        """Process entire dataset"""
        try:
            for split in ['train', 'test', 'valid']:
                data_logger.info(f"\nProcessing {split} split...")
                image_dir = self.data_dir / split / 'images'
                label_dir = self.data_dir / split / 'labels'
                
                if not image_dir.exists():
                    data_logger.warning(f"Warning: {image_dir} does not exist, skipping...")
                    continue
                
                image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
                for image_path in tqdm(image_files, desc=f"Augmenting {split} images"):
                    label_path = label_dir / f"{image_path.stem}.txt"
                    self.process_image_and_label(image_path, label_path, split)
        
        except Exception as e:
            data_logger.error(f"Error in dataset processing: {str(e)}")
            raise DataException(e, sys)

def main():
    # Define paths
    data_dir = Path("D:/External_Projects/Edith_Defene_System/birds_vs_drones_detection_and_tracking_version-0.0.1/data")
    output_dir = Path("D:/External_Projects/Edith_Defene_System/birds_vs_drones_detection_and_tracking_version-0.0.1/augmented_data")
    
    # Create and run augmentation pipeline
    augmenter = DataAugmentation(data_dir, output_dir)
    augmenter.process_dataset()
    
    data_logger.info("\nData augmentation completed!")
    data_logger.info(f"Augmented dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()
