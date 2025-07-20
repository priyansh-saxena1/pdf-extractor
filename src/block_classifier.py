"""
Block Classifier Module

This module implements the MiniLM-based block classifier for determining
heading levels (Title, H1-H4) of text regions.
"""

import os
import numpy as np
import onnxruntime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockClassifier:
    """
    Implements MiniLM-based block classifier for heading level detection.
    """
    
    def __init__(self, model_path='models/blockclf_quant.onnx', conf_threshold=0.55):
        """
        Initialize the block classifier with MiniLM ONNX model.
        
        Args:
            model_path (str): Path to the MiniLM ONNX model
            conf_threshold (float): Confidence threshold for classification
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Class mapping
        self.class_map = {
            0: "Title",
            1: "H1",
            2: "H2",
            3: "H3",
            4: "H4",
            5: "Other"
        }
        
        # Load ONNX model
        try:
            logger.info(f"Loading block classifier model from {model_path}")
            self.session = onnxruntime.InferenceSession(
                model_path, 
                providers=['CPUExecutionProvider']
            )
            
            # Get model metadata
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Block classifier model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load block classifier model: {e}")
            raise
    
    def preprocess_text(self, text, max_length=64):
        """
        Preprocess text for the MiniLM model input.
        This is a simplified tokenization for demonstration.
        
        Args:
            text (str): Input text
            max_length (int): Maximum token length
            
        Returns:
            numpy.ndarray: Preprocessed token ids
        """
        # Simple tokenization (in a real implementation, use a proper tokenizer)
        # This is a placeholder for the actual tokenization logic
        tokens = text.lower().split()[:max_length]
        
        # Convert to token ids (simplified)
        # In a real implementation, use a proper vocabulary lookup
        token_ids = np.zeros(max_length, dtype=np.int64)
        for i, token in enumerate(tokens[:max_length]):
            # Simple hash-based token ID generation (for demonstration only)
            token_ids[i] = hash(token) % 30000
            
        return token_ids
    
    def preprocess_features(self, features):
        """
        Preprocess features for the block classifier model.
        
        Args:
            features (dict): Features extracted from OCR
            
        Returns:
            dict: Preprocessed features
        """
        # Extract bbox features
        bbox_norm = np.array(features["bbox_norm"], dtype=np.float32)
        
        # Extract meta features
        meta_feats = np.array([
            features.get("area_ratio", 0.0),
            features.get("aspect_ratio", 0.0),
            features.get("y_center", 0.5),
            features.get("confidence", 0.0),
            0.0,  # font_ratio placeholder
            0.0   # whitespace_ratio placeholder
        ], dtype=np.float32)
        
        # Preprocess text
        token_ids = self.preprocess_text(features["text"])
        
        return {
            "token_ids": token_ids,
            "bbox_norm": bbox_norm,
            "meta_feats": meta_feats
        }
    
    def classify(self, features):
        """
        Classify a text block using the MiniLM model.
        
        Args:
            features (dict): Features extracted from OCR
            
        Returns:
            dict: Classification result with class and confidence
        """
        try:
            # Check if this is already a numbered heading
            if features.get("is_numbered_heading", False) and features.get("numbered_level"):
                return {
                    "class": features["numbered_level"],
                    "confidence": 1.0,
                    "is_heading": True,
                    "is_numbered": True
                }
            
            # Preprocess features
            preprocessed = self.preprocess_features(features)
            
            # Prepare inputs
            inputs = {
                "input_ids": np.expand_dims(preprocessed["token_ids"], axis=0),
                "bbox_features": np.expand_dims(preprocessed["bbox_norm"], axis=0),
                "meta_features": np.expand_dims(preprocessed["meta_feats"], axis=0)
            }
            
            # Run inference
            outputs = self.session.run([self.output_name], 
                                      {name: inputs[name.split(":")[-1]] for name in self.input_names})
            
            # Get class probabilities
            probs = outputs[0][0]
            
            # Get predicted class
            pred_class = np.argmax(probs)
            confidence = float(probs[pred_class])
            
            # Map to class name
            class_name = self.class_map[pred_class]
            
            # Check if confidence is above threshold
            is_heading = class_name != "Other" and confidence >= self.conf_threshold
            
            return {
                "class": class_name,
                "confidence": confidence,
                "is_heading": is_heading,
                "is_numbered": False
            }
            
        except Exception as e:
            logger.error(f"Block classification failed: {e}")
            return {
                "class": "Other",
                "confidence": 0.0,
                "is_heading": False,
                "is_numbered": False
            }
    
    def classify_blocks(self, ocr_features):
        """
        Classify multiple text blocks.
        
        Args:
            ocr_features (list): List of feature dictionaries from OCR
            
        Returns:
            list: List of classification results
        """
        results = []
        
        for features in ocr_features:
            classification = self.classify(features)
            
            # Combine OCR features with classification results
            result = {
                "text": features["text"],
                "bbox": features["bbox"],
                "level": classification["class"],
                "confidence": classification["confidence"],
                "is_heading": classification["is_heading"],
                "is_numbered": classification["is_numbered"]
            }
            
            results.append(result)
            
        return results 