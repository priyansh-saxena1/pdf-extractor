"""
Heading Detector Module

This module implements the YOLOv5-Nano detector for identifying heading regions in PDF pages.
"""

import os
import numpy as np
import onnxruntime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeadingDetector:
    """
    Implements YOLOv5-Nano detector for heading detection in PDF pages.
    """
    
    def __init__(self, model_path='models/yolo_nano.onnx', conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize the heading detector with YOLO model.
        
        Args:
            model_path (str): Path to the YOLO ONNX model
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for non-maximum suppression
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load ONNX model
        try:
            logger.info(f"Loading YOLO model from {model_path}")
            self.session = onnxruntime.InferenceSession(
                model_path, 
                providers=['CPUExecutionProvider']
            )
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def preprocess(self, image_array):
        """
        Preprocess the image for YOLO model input.
        
        Args:
            image_array (numpy.ndarray): Input image as numpy array (CHW format)
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Ensure the input is float32 and normalized to [0, 1]
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32)
            
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
            
        # Add batch dimension
        preprocessed = np.expand_dims(image_array, axis=0)
        
        return preprocessed
    
    def detect(self, image_array):
        """
        Detect heading regions in an image.
        
        Args:
            image_array (numpy.ndarray): Input image as numpy array (CHW format)
            
        Returns:
            list: List of detected heading regions as [x0, y0, x1, y1, confidence]
        """
        try:
            # Preprocess the image
            input_tensor = self.preprocess(image_array)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Process the outputs (depends on YOLOv5 export format)
            # For YOLOv5 ONNX models, the output is typically [batch, num_detections, 5+num_classes]
            # where 5 represents [x, y, w, h, confidence]
            detections = self._process_output(outputs[0])
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _process_output(self, output):
        """
        Process YOLO output to extract bounding boxes, confidence scores, and apply NMS.
        
        Args:
            output (numpy.ndarray): Raw output from YOLO model
            
        Returns:
            list: List of filtered detections as [x0, y0, x1, y1, confidence]
        """
        # Extract detections above confidence threshold
        # Assuming output format is [batch, num_detections, 5+num_classes]
        # where 5 represents [x, y, w, h, confidence]
        
        # Get detections above threshold
        valid_detections = output[output[:, :, 4] > self.conf_threshold]
        
        if len(valid_detections) == 0:
            return []
        
        # Convert from [x, y, w, h] to [x0, y0, x1, y1]
        boxes = []
        for detection in valid_detections[0]:
            confidence = float(detection[4])
            
            if confidence < self.conf_threshold:
                continue
                
            # Extract box coordinates
            x, y, w, h = detection[0:4]
            
            # Convert from center coordinates to top-left, bottom-right
            x0 = float(x - w/2)
            y0 = float(y - h/2)
            x1 = float(x + w/2)
            y1 = float(y + h/2)
            
            # Add to boxes
            boxes.append([x0, y0, x1, y1, confidence])
        
        # Apply non-maximum suppression
        return self._non_max_suppression(boxes)
    
    def _non_max_suppression(self, boxes):
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            boxes (list): List of detections as [x0, y0, x1, y1, confidence]
            
        Returns:
            list: Filtered list of detections
        """
        if not boxes:
            return []
            
        # Sort by confidence
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        
        # Apply NMS
        keep = []
        while boxes:
            keep.append(boxes[0])
            boxes = [box for box in boxes[1:] if self._iou(box, keep[-1]) < self.iou_threshold]
            
        return keep
    
    def _iou(self, box1, box2):
        """
        Calculate IoU between two boxes.
        
        Args:
            box1 (list): First box as [x0, y0, x1, y1, confidence]
            box2 (list): Second box as [x0, y0, x1, y1, confidence]
            
        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x0_inter = max(box1[0], box2[0])
        y0_inter = max(box1[1], box2[1])
        x1_inter = min(box1[2], box2[2])
        y1_inter = min(box1[3], box2[3])
        
        if x1_inter < x0_inter or y1_inter < y0_inter:
            return 0.0
            
        inter_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        return iou 