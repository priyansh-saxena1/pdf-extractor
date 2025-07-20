"""
OCR Processor Module

This module handles OCR processing using Tesseract for text recognition in detected regions.
"""

import os
import numpy as np
import pytesseract
import logging
from PIL import Image
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    Handles OCR processing using Tesseract for text recognition.
    """
    
    def __init__(self, config='--oem 1 --psm 6'):
        """
        Initialize the OCR processor.
        
        Args:
            config (str): Tesseract configuration string
        """
        self.config = config
        
        # Regex pattern for numbered headings
        self.heading_pattern = re.compile(r'^\s*(\d+(?:\.\d+)*)\s+(.*)')
        
    def process_image(self, image, lang='eng'):
        """
        Process an image with Tesseract OCR.
        
        Args:
            image (PIL.Image): Input image
            lang (str): Language code for Tesseract
            
        Returns:
            dict: OCR results with text and bounding boxes
        """
        try:
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Run OCR with pytesseract
            ocr_data = pytesseract.image_to_data(
                image, 
                lang=lang,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            return self._process_ocr_data(ocr_data)
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {"text": "", "lines": []}
    
    def _process_ocr_data(self, ocr_data):
        """
        Process raw OCR data from Tesseract.
        
        Args:
            ocr_data (dict): Raw OCR data from pytesseract
            
        Returns:
            dict: Processed OCR results with text and lines
        """
        # Extract text and confidence from OCR data
        text_blocks = []
        current_line = []
        current_line_num = -1
        full_text = ""
        
        # Process OCR data by line
        for i in range(len(ocr_data['text'])):
            # Skip empty text
            if not ocr_data['text'][i].strip():
                continue
                
            # Check if this is a new line
            if ocr_data['line_num'][i] != current_line_num:
                # Save previous line if it exists
                if current_line:
                    line_text = " ".join([block["text"] for block in current_line])
                    text_blocks.append({
                        "text": line_text,
                        "bbox": self._merge_bboxes([block["bbox"] for block in current_line]),
                        "confidence": sum(block["confidence"] for block in current_line) / len(current_line)
                    })
                    full_text += line_text + "\n"
                
                # Start new line
                current_line = []
                current_line_num = ocr_data['line_num'][i]
            
            # Add current word to line
            current_line.append({
                "text": ocr_data['text'][i],
                "bbox": (
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['left'][i] + ocr_data['width'][i],
                    ocr_data['top'][i] + ocr_data['height'][i]
                ),
                "confidence": float(ocr_data['conf'][i]) / 100.0 if ocr_data['conf'][i] > 0 else 0.0
            })
        
        # Add the last line if it exists
        if current_line:
            line_text = " ".join([block["text"] for block in current_line])
            text_blocks.append({
                "text": line_text,
                "bbox": self._merge_bboxes([block["bbox"] for block in current_line]),
                "confidence": sum(block["confidence"] for block in current_line) / len(current_line)
            })
            full_text += line_text + "\n"
        
        return {
            "text": full_text.strip(),
            "lines": text_blocks
        }
    
    def _merge_bboxes(self, bboxes):
        """
        Merge multiple bounding boxes into a single bounding box.
        
        Args:
            bboxes (list): List of bounding boxes as (x0, y0, x1, y1)
            
        Returns:
            tuple: Merged bounding box as (x0, y0, x1, y1)
        """
        x0 = min(bbox[0] for bbox in bboxes)
        y0 = min(bbox[1] for bbox in bboxes)
        x1 = max(bbox[2] for bbox in bboxes)
        y1 = max(bbox[3] for bbox in bboxes)
        
        return (x0, y0, x1, y1)
    
    def check_for_numbered_heading(self, text):
        """
        Check if text contains a numbered heading pattern and extract level.
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (bool, str, int) - (is_heading, cleaned_text, level)
        """
        match = self.heading_pattern.match(text)
        
        if match:
            number_part = match.group(1)
            text_part = match.group(2)
            
            # Determine level based on number of dots
            level = number_part.count('.') + 1
            level_str = f"H{min(level, 4)}"  # Cap at H4
            
            return True, text_part.strip(), level_str
        
        return False, text, None
    
    def extract_features(self, ocr_result, page_width, page_height):
        """
        Extract features from OCR results for block classification.
        
        Args:
            ocr_result (dict): OCR results
            page_width (int): Width of the page
            page_height (int): Height of the page
            
        Returns:
            list: List of feature dictionaries for each text line
        """
        features = []
        
        for line in ocr_result.get("lines", []):
            # Extract bounding box
            x0, y0, x1, y1 = line["bbox"]
            
            # Normalize coordinates
            x0_norm = x0 / page_width
            y0_norm = y0 / page_height
            x1_norm = x1 / page_width
            y1_norm = y1 / page_height
            
            # Calculate additional features
            width = x1 - x0
            height = y1 - y0
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # Calculate position features
            y_center = (y0 + y1) / 2 / page_height
            
            # Check for numbered heading
            is_heading, text, level = self.check_for_numbered_heading(line["text"])
            
            features.append({
                "text": text,
                "bbox": (x0, y0, x1, y1),
                "bbox_norm": (x0_norm, y0_norm, x1_norm, y1_norm),
                "confidence": line["confidence"],
                "area_ratio": area / (page_width * page_height),
                "aspect_ratio": aspect_ratio,
                "y_center": y_center,
                "is_numbered_heading": is_heading,
                "numbered_level": level
            })
        
        return features 