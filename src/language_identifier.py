"""
Language Identifier Module

This module implements language identification for text regions using a FastText-based
ONNX model to determine the script/language of text.
"""

import os
import numpy as np
import onnxruntime
import logging
import string
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageIdentifier:
    """
    Implements language/script identification using a FastText-based ONNX model.
    """
    
    def __init__(self, model_path='models/langid_quant.onnx'):
        """
        Initialize the language identifier with FastText ONNX model.
        
        Args:
            model_path (str): Path to the FastText ONNX model
        """
        self.model_path = model_path
        
        # Language/script mapping
        self.language_map = {
            0: "eng",       # English/Latin script
            1: "chi_sim",   # Simplified Chinese
            2: "jpn",       # Japanese
            3: "ara",       # Arabic
            4: "hin",       # Hindi/Devanagari
            5: "other"      # Other scripts
        }
        
        # Tesseract language pack mapping
        self.tesseract_lang_map = {
            "eng": "eng",
            "chi_sim": "chi_sim",
            "jpn": "jpn",
            "ara": "ara",
            "hin": "hin",
            "other": "eng"  # Default to English for other scripts
        }
        
        # Load ONNX model
        try:
            logger.info(f"Loading language identification model from {model_path}")
            self.session = onnxruntime.InferenceSession(
                model_path, 
                providers=['CPUExecutionProvider']
            )
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"Language identification model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load language identification model: {e}")
            raise
    
    def preprocess_text(self, text, max_length=300):
        """
        Preprocess text for the FastText model input.
        
        Args:
            text (str): Input text
            max_length (int): Maximum number of bytes to process
            
        Returns:
            numpy.ndarray: Preprocessed text as byte array
        """
        # Convert to lowercase and remove excessive whitespace
        text = text.lower().strip()
        
        # Truncate to max_length bytes
        text_bytes = text.encode('utf-8', errors='ignore')[:max_length]
        
        # Convert to numpy array of byte values
        byte_array = np.frombuffer(text_bytes, dtype=np.uint8)
        
        # Pad or truncate to fixed length
        padded = np.zeros(max_length, dtype=np.float32)
        padded[:len(byte_array)] = byte_array / 255.0  # Normalize to [0,1]
        
        return padded
    
    def identify_language(self, text):
        """
        Identify the language/script of the given text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Identified language code for Tesseract
        """
        try:
            # If text is too short, default to English
            if len(text) < 5:
                return self.tesseract_lang_map["eng"]
            
            # Preprocess the text
            input_tensor = self.preprocess_text(text)
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            
            # Get the predicted language class
            pred_class = np.argmax(outputs[0][0])
            lang_code = self.language_map[pred_class]
            
            return self.tesseract_lang_map[lang_code]
            
        except Exception as e:
            logger.error(f"Language identification failed: {e}")
            return self.tesseract_lang_map["eng"]  # Default to English on error
    
    def identify_language_from_ocr(self, ocr_result):
        """
        Identify the dominant language from OCR results.
        
        Args:
            ocr_result (dict): OCR result containing text blocks
            
        Returns:
            str: Identified language code for Tesseract
        """
        # Extract all text from OCR result
        all_text = ""
        
        # Structure depends on OCR output format
        if isinstance(ocr_result, dict) and "text" in ocr_result:
            all_text = ocr_result["text"]
        elif isinstance(ocr_result, list):
            all_text = " ".join([block.get("text", "") for block in ocr_result if "text" in block])
        
        return self.identify_language(all_text)
    
    def get_script_confidence(self, text):
        """
        Get confidence scores for each supported script.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary mapping script names to confidence scores
        """
        try:
            # Preprocess the text
            input_tensor = self.preprocess_text(text)
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            
            # Get confidence scores for each language
            scores = outputs[0][0]
            
            # Create dictionary of script to confidence
            confidence = {self.language_map[i]: float(score) for i, score in enumerate(scores)}
            
            return confidence
            
        except Exception as e:
            logger.error(f"Script confidence calculation failed: {e}")
            return {lang: 0.0 for lang in self.language_map.values()} 