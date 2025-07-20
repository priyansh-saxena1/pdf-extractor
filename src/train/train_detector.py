#!/usr/bin/env python3
"""
YOLO Heading Detector Training Script

This script trains a YOLOv5-Nano model for heading detection in PDF pages.
It uses PyTorch and the Ultralytics YOLOv5 API.
"""

import os
import sys
import argparse
import logging
import torch
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HeadingDetectorTrainer:
    """
    Trainer for YOLOv5-Nano heading detector model.
    """
    
    def __init__(self, data_yaml, output_dir='models', epochs=100, batch_size=16, img_size=640):
        """
        Initialize the heading detector trainer.
        
        Args:
            data_yaml (str): Path to the data YAML file
            output_dir (str): Directory to save the trained model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            img_size (int): Input image size for the model
        """
        self.data_yaml = data_yaml
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self):
        """
        Train the YOLOv5-Nano model.
        """
        try:
            # Import YOLOv5 modules (assuming YOLOv5 is installed)
            # In a real implementation, you would have YOLOv5 as a dependency
            from yolov5 import train
            
            # Set up training arguments
            args = {
                'data': self.data_yaml,
                'weights': 'yolov5n.pt',  # Start from YOLOv5-Nano pretrained weights
                'cfg': '',  # Use default config
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'img_size': [self.img_size],
                'rect': False,
                'resume': False,
                'nosave': False,
                'noval': False,
                'noautoanchor': False,
                'evolve': False,
                'bucket': '',
                'cache_images': False,
                'name': 'heading_detector',
                'device': '',  # Auto-select
                'multi_scale': False,
                'single_cls': False,
                'adam': False,
                'sync_bn': False,
                'local_rank': -1,
                'project': self.output_dir,
                'entity': None,
                'exist_ok': True,
                'quad': False,
                'linear_lr': False,
                'label_smoothing': 0.0,
                'upload_dataset': False,
                'bbox_interval': -1,
                'save_period': -1,
                'artifact_alias': 'latest'
            }
            
            # Start training
            logger.info(f"Starting YOLOv5-Nano training for heading detection...")
            train.run(**args)
            
            # Export to ONNX
            self.export_to_onnx()
            
            logger.info(f"Training completed successfully.")
            
        except ImportError:
            logger.error("YOLOv5 is not installed. Please install it first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)
    
    def export_to_onnx(self):
        """
        Export the trained model to ONNX format.
        """
        try:
            # Import YOLOv5 export module
            from yolov5 import export
            
            # Path to the best trained model
            weights = os.path.join(self.output_dir, 'heading_detector', 'weights', 'best.pt')
            
            # Output ONNX path
            onnx_path = os.path.join(self.output_dir, 'yolo_nano.onnx')
            
            # Export arguments
            args = {
                'weights': weights,
                'img_size': [self.img_size, self.img_size],
                'batch_size': 1,
                'device': 'cpu',
                'include': ['onnx'],
                'half': False,
                'inplace': False,
                'train': False,
                'optimize': True,
                'dynamic': True,
                'simplify': True,
                'opset': 11
            }
            
            logger.info(f"Exporting model to ONNX format...")
            export.run(**args)
            
            # Rename the output file if needed
            if os.path.exists(weights.replace('.pt', '.onnx')):
                os.rename(weights.replace('.pt', '.onnx'), onnx_path)
                
            logger.info(f"Model exported to {onnx_path}")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    
    def create_data_yaml(self, train_path, val_path, num_classes=1):
        """
        Create a data YAML file for YOLOv5 training.
        
        Args:
            train_path (str): Path to training images
            val_path (str): Path to validation images
            num_classes (int): Number of classes
            
        Returns:
            str: Path to the created YAML file
        """
        # Create data configuration
        data = {
            'train': train_path,
            'val': val_path,
            'nc': num_classes,
            'names': ['heading']
        }
        
        # Write to YAML file
        yaml_path = os.path.join(self.output_dir, 'heading_data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
        return yaml_path


def main():
    """
    Main entry point for the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv5-Nano heading detector')
    parser.add_argument('--data', required=True, help='Path to data YAML file')
    parser.add_argument('--output-dir', default='models', help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    args = parser.parse_args()
    
    # Create and run the trainer
    trainer = HeadingDetectorTrainer(
        data_yaml=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    trainer.train()


if __name__ == "__main__":
    main() 