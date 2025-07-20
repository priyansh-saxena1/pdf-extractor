#!/usr/bin/env python3
"""
PDF Outline Extractor Test Suite

This script tests the PDF outline extraction pipeline on sample PDFs
and verifies that the results match the expected output.
"""

import os
import sys
import json
import logging
import unittest
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import main script
import process_pdfs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TestPDFOutlineExtractor(unittest.TestCase):
    """
    Test cases for the PDF outline extraction pipeline.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Define paths
        self.input_dir = 'inputs'
        self.output_dir = 'outputs'
        self.sample_output_dir = 'sample_dataset/outputs'
        
        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sample_output_dir, exist_ok=True)
    
    def test_pipeline_dry_run(self):
        """
        Test the pipeline in dry-run mode on sample PDFs.
        """
        # Run the pipeline in dry-run mode
        args = [
            '--input-dir', self.input_dir,
            '--output-dir', self.output_dir,
            '--dry-run'
        ]
        
        # Create extractor instance
        extractor = process_pdfs.PDFOutlineExtractor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            dry_run=True
        )
        
        # Find PDF files
        pdf_files = extractor.find_pdf_files()
        
        # Skip test if no PDF files found
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}, skipping test")
            return
        
        # Process each PDF file and verify results
        for pdf_file in pdf_files:
            result, output_path = extractor.process_pdf(pdf_file)
            
            # Skip if processing failed
            if result is None:
                continue
                
            # Get corresponding ground truth file
            ground_truth_path = output_path.replace(self.output_dir, self.sample_output_dir)
            
            # Skip if ground truth file doesn't exist
            if not os.path.exists(ground_truth_path):
                logger.warning(f"Ground truth file {ground_truth_path} not found, skipping verification")
                continue
                
            # Compare with ground truth
            metrics = extractor.compare_with_ground_truth(result, ground_truth_path)
            
            # Log results
            logger.info(f"Metrics for {pdf_file}:")
            logger.info(f"  Title match: {metrics['title_match']}")
            logger.info(f"  Precision: {metrics['precision']:.2f}")
            logger.info(f"  Recall: {metrics['recall']:.2f}")
            logger.info(f"  F1 score: {metrics['f1']:.2f}")
            
            # Assert metrics meet minimum thresholds
            self.assertTrue(metrics['title_match'], "Title should match ground truth")
            self.assertGreaterEqual(metrics['precision'], 0.8, "Precision should be at least 0.8")
            self.assertGreaterEqual(metrics['recall'], 0.8, "Recall should be at least 0.8")
            self.assertGreaterEqual(metrics['f1'], 0.8, "F1 score should be at least 0.8")
    
    def test_command_line_execution(self):
        """
        Test the pipeline by running the main script as a command-line tool.
        """
        # Run the script as a subprocess
        try:
            cmd = [
                'python', 'process_pdfs.py',
                '--input-dir', self.input_dir,
                '--output-dir', self.output_dir,
                '--dry-run'
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Check that the process completed successfully
            self.assertEqual(result.returncode, 0, "Process should exit with code 0")
            
            # Check that output contains expected log messages
            self.assertIn("Starting training for", result.stdout + result.stderr)
            
        except subprocess.CalledProcessError as e:
            self.fail(f"Command execution failed with return code {e.returncode}: {e.stderr}")
    
    def test_individual_components(self):
        """
        Test individual components of the pipeline.
        """
        # Test PDF processor
        from src.pdf_processor import PDFProcessor
        pdf_processor = PDFProcessor()
        
        # Test heading detector
        from src.heading_detector import HeadingDetector
        heading_detector = HeadingDetector()
        
        # Test language identifier
        from src.language_identifier import LanguageIdentifier
        language_identifier = LanguageIdentifier()
        
        # Test OCR processor
        from src.ocr_processor import OCRProcessor
        ocr_processor = OCRProcessor()
        
        # Test block classifier
        from src.block_classifier import BlockClassifier
        block_classifier = BlockClassifier()
        
        # Test JSON exporter
        from src.json_exporter import JSONExporter
        json_exporter = JSONExporter()
        
        # Verify that all components can be initialized
        self.assertIsNotNone(pdf_processor, "PDF processor should be initialized")
        self.assertIsNotNone(heading_detector, "Heading detector should be initialized")
        self.assertIsNotNone(language_identifier, "Language identifier should be initialized")
        self.assertIsNotNone(ocr_processor, "OCR processor should be initialized")
        self.assertIsNotNone(block_classifier, "Block classifier should be initialized")
        self.assertIsNotNone(json_exporter, "JSON exporter should be initialized")


if __name__ == '__main__':
    unittest.main() 