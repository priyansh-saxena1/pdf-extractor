"""
PDF Processor Module

This module handles the extraction of text and rendering of PDF pages
using PyMuPDF (fitz).
"""

import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF processing operations including text extraction and page rendering.
    """
    
    def __init__(self, dpi=300, target_size=(640, 640)):
        """
        Initialize the PDF processor.
        
        Args:
            dpi (int): DPI for rendering PDF pages
            target_size (tuple): Target size for the rendered images (width, height)
        """
        self.dpi = dpi
        self.target_size = target_size
        self.zoom = dpi / 72  # PDF standard is 72 dpi
        
    def open_pdf(self, pdf_path):
        """
        Open a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            fitz.Document: The opened PDF document
        """
        try:
            return fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            raise
    
    def extract_page_text(self, page):
        """
        Extract text spans from a PDF page.
        
        Args:
            page (fitz.Page): A PDF page
            
        Returns:
            dict: Dictionary with text spans and their properties
        """
        try:
            return page.get_text("dict")
        except Exception as e:
            logger.error(f"Failed to extract text from page {page.number}: {e}")
            return {"blocks": []}
    
    def render_page(self, page):
        """
        Render a PDF page as a PIL image.
        
        Args:
            page (fitz.Page): A PDF page
            
        Returns:
            PIL.Image: The rendered page as a PIL image
        """
        try:
            # Create a pixmap with RGB color space
            matrix = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PIL image
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # Resize to target size
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.LANCZOS)
                
            return img
            
        except Exception as e:
            logger.error(f"Failed to render page {page.number}: {e}")
            raise
    
    def get_page_count(self, pdf_doc):
        """
        Get the number of pages in a PDF document.
        
        Args:
            pdf_doc (fitz.Document): A PDF document
            
        Returns:
            int: Number of pages
        """
        return len(pdf_doc)
    
    def extract_metadata(self, pdf_doc):
        """
        Extract metadata from a PDF document.
        
        Args:
            pdf_doc (fitz.Document): A PDF document
            
        Returns:
            dict: PDF metadata
        """
        return pdf_doc.metadata
    
    def image_to_array(self, image):
        """
        Convert a PIL image to a numpy array for model input.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            numpy.ndarray: Image as numpy array in RGB format
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Transpose from HWC to CHW format for ONNX models
        img_array = img_array.transpose(2, 0, 1)
        
        return img_array
    
    def crop_image(self, image, bbox):
        """
        Crop a region from an image based on bounding box.
        
        Args:
            image (PIL.Image): Input image
            bbox (tuple): Bounding box coordinates (x0, y0, x1, y1)
            
        Returns:
            PIL.Image: Cropped image
        """
        x0, y0, x1, y1 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        width, height = image.size
        x0 = max(0, min(x0, width-1))
        y0 = max(0, min(y0, height-1))
        x1 = max(x0+1, min(x1, width))
        y1 = max(y0+1, min(y1, height))
        
        return image.crop((x0, y0, x1, y1)) 