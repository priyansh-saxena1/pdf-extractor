#!/usr/bin/env python3
"""
PDF Outline Extractor

This script processes PDF files to extract outlines (headings and titles)
using a vision-first, multilingual approach. It integrates multiple models:
- YOLOv5-Nano for heading detection
- FastText for language identification
- Tesseract OCR for text recognition
- MiniLM for block classification

The script processes all PDFs in the input directory and generates
JSON output files according to the specified schema.
"""

import os
import sys
import time
import argparse
import logging
import json
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import custom modules
from src.pdf_processor import PDFProcessor
from src.heading_detector import HeadingDetector
from src.language_identifier import LanguageIdentifier
from src.ocr_processor import OCRProcessor
from src.block_classifier import BlockClassifier
from src.json_exporter import JSONExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    """
    Main class for PDF outline extraction pipeline.
    """
    
    def __init__(self, input_dir='inputs', output_dir='outputs', model_dir='models',
                 max_workers=8, dry_run=False):
        """
        Initialize the PDF outline extractor.
        
        Args:
            input_dir (str): Directory containing input PDF files
            output_dir (str): Directory for output JSON files
            model_dir (str): Directory containing model files
            max_workers (int): Maximum number of worker threads
            dry_run (bool): If True, compare results with ground truth
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.max_workers = max_workers
        self.dry_run = dry_run
        
        # Create directories if they don't exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        self.pdf_processor = PDFProcessor(dpi=300, target_size=(640, 640))
        self.heading_detector = HeadingDetector(model_path=os.path.join(model_dir, 'yolo_nano.onnx'))
        self.language_identifier = LanguageIdentifier(model_path=os.path.join(model_dir, 'langid_quant.onnx'))
        self.ocr_processor = OCRProcessor(config='--oem 1 --psm 6')
        self.block_classifier = BlockClassifier(model_path=os.path.join(model_dir, 'blockclf_quant.onnx'))
        self.json_exporter = JSONExporter(schema_path='schema/output_schema.json')
        
    def find_pdf_files(self):
        """
        Find all PDF files in the input directory.
        
        Returns:
            list: List of paths to PDF files
        """
        pdf_pattern = os.path.join(self.input_dir, '*.pdf')
        pdf_files = glob.glob(pdf_pattern)
        
        logger.info(f"Found {len(pdf_files)} PDF files in {self.input_dir}")
        return pdf_files
    
    def process_page(self, page, page_num):
        """
        Process a single PDF page.
        
        Args:
            page (fitz.Page): PDF page to process
            page_num (int): Page number (0-indexed)
            
        Returns:
            dict: Dictionary with page processing results
        """
        # Extract text spans
        page_text = self.pdf_processor.extract_page_text(page)
        
        # Render page as image
        page_image = self.pdf_processor.render_page(page)
        page_width, page_height = page_image.size
        
        # Convert image to array for model input
        image_array = self.pdf_processor.image_to_array(page_image)
        
        # Detect heading regions
        detections = self.heading_detector.detect(image_array)
        
        # Process each detected region
        region_results = []
        
        for detection in detections:
            x0, y0, x1, y1, confidence = detection
            
            # Scale coordinates to original image size
            x0 = int(x0 * page_width / 640)
            y0 = int(y0 * page_height / 640)
            x1 = int(x1 * page_width / 640)
            y1 = int(y1 * page_height / 640)
            
            # Crop the region
            region_image = self.pdf_processor.crop_image(page_image, (x0, y0, x1, y1))
            
            # Identify language
            # For simplicity, we'll use English as default
            # In a real implementation, extract text from the region and identify language
            lang = 'eng'
            
            # Perform OCR on the region
            ocr_result = self.ocr_processor.process_image(region_image, lang=lang)
            
            # Extract features for classification
            ocr_features = self.ocr_processor.extract_features(ocr_result, page_width, page_height)
            
            # Classify each text line
            classifications = self.block_classifier.classify_blocks(ocr_features)
            
            # Add to results
            for classification in classifications:
                if classification["is_heading"]:
                    region_results.append({
                        "text": classification["text"],
                        "level": classification["level"],
                        "confidence": classification["confidence"],
                        "bbox": classification["bbox"],
                        "page": page_num + 1  # Convert to 1-indexed
                    })
        
        return {
            "page_num": page_num,
            "regions": region_results
        }
    
    def process_pdf(self, pdf_path):
        """
        Process a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Extracted outline data
        """
        try:
            start_time = time.time()
            logger.info(f"Processing {pdf_path}...")
            
            # Open PDF
            pdf_doc = self.pdf_processor.open_pdf(pdf_path)
            num_pages = self.pdf_processor.get_page_count(pdf_doc)
            
            # Extract metadata
            metadata = self.pdf_processor.extract_metadata(pdf_doc)
            
            # Process pages in parallel
            all_regions = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_page = {
                    executor.submit(self.process_page, pdf_doc[page_num], page_num): page_num
                    for page_num in range(num_pages)
                }
                
                for future in as_completed(future_to_page):
                    page_result = future.result()
                    all_regions.extend(page_result["regions"])
            
            # Sort regions by page and y-position
            all_regions.sort(key=lambda r: (r["page"], r["bbox"][1]))
            
            # Extract title (highest confidence Title on first page)
            title_candidates = [
                r for r in all_regions 
                if r["level"] == "Title" and r["page"] == 1 and r["confidence"] > 0.9
            ]
            
            title = metadata.get("title", "")
            if title_candidates:
                title_candidates.sort(key=lambda r: r["confidence"], reverse=True)
                title = title_candidates[0]["text"]
            
            # Format results
            result = self.json_exporter.format_results(title, all_regions)
            
            # Generate output path
            output_path = self.json_exporter.generate_output_path(pdf_path, self.output_dir)
            
            # Export to JSON
            if not self.dry_run:
                self.json_exporter.export_to_json(result, output_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Processed {pdf_path} in {elapsed_time:.2f} seconds")
            
            return result, output_path
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            return None, None
    
    def compare_with_ground_truth(self, result, ground_truth_path):
        """
        Compare results with ground truth for dry run.
        
        Args:
            result (dict): Extracted outline data
            ground_truth_path (str): Path to ground truth JSON file
            
        Returns:
            dict: Comparison metrics
        """
        try:
            # Load ground truth
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
                
            # Compare title
            title_match = result["title"].lower() == ground_truth["title"].lower()
            
            # Compare outline items
            result_items = set((item["level"], item["text"].lower(), item["page"]) 
                              for item in result["outline"])
            truth_items = set((item["level"], item["text"].lower(), item["page"]) 
                             for item in ground_truth["outline"])
            
            # Calculate metrics
            common_items = result_items.intersection(truth_items)
            precision = len(common_items) / len(result_items) if result_items else 0
            recall = len(common_items) / len(truth_items) if truth_items else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # List mismatches
            false_positives = result_items - truth_items
            false_negatives = truth_items - result_items
            
            return {
                "title_match": title_match,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": list(false_positives),
                "false_negatives": list(false_negatives)
            }
            
        except Exception as e:
            logger.error(f"Failed to compare with ground truth {ground_truth_path}: {e}")
            return {
                "title_match": False,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "error": str(e)
            }
    
    def run(self):
        """
        Run the PDF outline extraction pipeline on all PDF files.
        """
        # Find PDF files
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return
        
        # Process PDFs in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pdf = {
                executor.submit(self.process_pdf, pdf_path): pdf_path
                for pdf_path in pdf_files
            }
            
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result, output_path = future.result()
                    
                    if result is not None:
                        # If dry run, compare with ground truth
                        if self.dry_run:
                            ground_truth_path = output_path.replace(self.output_dir, 'sample_dataset/outputs')
                            if os.path.exists(ground_truth_path):
                                metrics = self.compare_with_ground_truth(result, ground_truth_path)
                                logger.info(f"Comparison metrics for {pdf_path}:")
                                logger.info(f"  Title match: {metrics['title_match']}")
                                logger.info(f"  Precision: {metrics['precision']:.2f}")
                                logger.info(f"  Recall: {metrics['recall']:.2f}")
                                logger.info(f"  F1 score: {metrics['f1']:.2f}")
                        
                        results.append((pdf_path, result))
                        
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")
        
        logger.info(f"Successfully processed {len(results)} out of {len(pdf_files)} PDF files")


def main():
    """
    Main entry point for the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract outlines from PDF files')
    parser.add_argument('--input-dir', default='inputs', help='Input directory containing PDF files')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for JSON files')
    parser.add_argument('--model-dir', default='models', help='Directory containing model files')
    parser.add_argument('--max-workers', type=int, default=8, help='Maximum number of worker threads')
    parser.add_argument('--dry-run', action='store_true', help='Compare results with ground truth')
    args = parser.parse_args()
    
    # Create and run the extractor
    extractor = PDFOutlineExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        max_workers=args.max_workers,
        dry_run=args.dry_run
    )
    
    # Measure total execution time
    start_time = time.time()
    extractor.run()
    elapsed_time = time.time() - start_time
    
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main() 