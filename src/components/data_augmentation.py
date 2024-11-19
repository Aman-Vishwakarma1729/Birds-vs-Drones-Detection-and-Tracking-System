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
            
            # Define augmentation pipeline with aerial imagery specific transformations
            self.transform = A.Compose([
                # Color augmentations for different lighting conditions
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                ], p=0.8),
                
                # Noise and blur for different weather and atmospheric conditions
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    A.MedianBlur(blur_limit=5, p=0.5),
                ], p=0.5),
                
                # Geometric transformations for different viewing angles
                A.OneOf([
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
                    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-45, 45), shear=(-10, 10), p=0.5),
                ], p=0.8),
                
                # Weather and atmospheric effects
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, p=0.3),
                    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.3),
                    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.3),
                ], p=0.5),
                
                # Additional transformations for robustness
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                ], p=0.5),
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
            total_images = 0
            processed_images = 0
            
            # Count total images first
            for split in ['train', 'test', 'valid']:
                image_dir = self.data_dir / split / 'images'
                if image_dir.exists():
                    total_images += len(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
            
            # Process each split
            with tqdm(total=total_images, desc="Overall Progress") as pbar:
                for split in ['train', 'test', 'valid']:
                    data_logger.info(f"\nProcessing {split} split...")
                    image_dir = self.data_dir / split / 'images'
                    label_dir = self.data_dir / split / 'labels'
                    
                    if not image_dir.exists():
                        data_logger.warning(f"Warning: {image_dir} does not exist, skipping...")
                        continue
                    
                    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
                    for image_path in image_files:
                        label_path = label_dir / f"{image_path.stem}.txt"
                        self.process_image_and_label(image_path, label_path, split)
                        processed_images += 1
                        pbar.update(1)
            
            data_logger.info(f"\nProcessed {processed_images} images successfully")
        
        except Exception as e:
            data_logger.error(f"Error in dataset processing: {str(e)}")
            raise DataException(e, sys)

def main():
    try:
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        
        # Define paths relative to project root
        data_dir = project_root / "data"
        output_dir = project_root / "augmented_data"
        
        # Validate directories
        if not data_dir.exists():
            raise DataException(f"Data directory not found: {data_dir}", sys)
        
        data_logger.info(f"Starting data augmentation pipeline...")
        data_logger.info(f"Data directory: {data_dir}")
        data_logger.info(f"Output directory: {output_dir}")
        
        # Create and run augmentation pipeline
        augmenter = DataAugmentation(data_dir, output_dir)
        augmenter.process_dataset()
        
        data_logger.info("\nData augmentation completed successfully!")
        data_logger.info(f"Augmented dataset saved to: {output_dir}")
        
    except Exception as e:
        data_logger.error(f"Error in main: {str(e)}")
        raise DataException(e, sys)

if __name__ == "__main__":
    main()
