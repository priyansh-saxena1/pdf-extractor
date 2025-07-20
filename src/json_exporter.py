"""
JSON Exporter Module

This module handles formatting and exporting the extracted outline data to JSON
according to the specified schema.
"""

import os
import json
import logging
import jsonschema
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JSONExporter:
    """
    Handles formatting and exporting of extracted outline data to JSON.
    """
    
    def __init__(self, schema_path='schema/output_schema.json'):
        """
        Initialize the JSON exporter.
        
        Args:
            schema_path (str): Path to the JSON schema file
        """
        self.schema_path = schema_path
        self.schema = self._load_schema(schema_path)
        
    def _load_schema(self, schema_path):
        """
        Load the JSON schema from file.
        
        Args:
            schema_path (str): Path to the JSON schema file
            
        Returns:
            dict: Loaded JSON schema
        """
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            return schema
        except Exception as e:
            logger.error(f"Failed to load JSON schema from {schema_path}: {e}")
            # Return a minimal schema if loading fails
            return {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "outline": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "level": {"type": "string"},
                                "text": {"type": "string"},
                                "page": {"type": "integer"}
                            },
                            "required": ["level", "text", "page"]
                        }
                    }
                },
                "required": ["title", "outline"]
            }
    
    def format_results(self, title, outline_items):
        """
        Format the extracted data according to the schema.
        
        Args:
            title (str): Document title
            outline_items (list): List of outline items
            
        Returns:
            dict: Formatted data
        """
        # Format the outline items
        formatted_items = []
        
        for item in outline_items:
            formatted_item = {
                "level": item["level"],
                "text": item["text"],
                "page": item["page"]
            }
            formatted_items.append(formatted_item)
        
        # Create the result structure
        result = {
            "title": title,
            "outline": formatted_items
        }
        
        return result
    
    def validate_against_schema(self, data):
        """
        Validate the data against the JSON schema.
        
        Args:
            data (dict): Data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"JSON validation error: {e}")
            return False
    
    def export_to_json(self, data, output_path):
        """
        Export data to a JSON file.
        
        Args:
            data (dict): Data to export
            output_path (str): Path to the output JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Validate data against schema
            if not self.validate_against_schema(data):
                logger.error(f"Data does not conform to schema, export aborted")
                return False
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Successfully exported JSON to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export JSON to {output_path}: {e}")
            return False
    
    def generate_output_path(self, input_path, output_dir):
        """
        Generate the output path for a given input path.
        
        Args:
            input_path (str): Path to the input file
            output_dir (str): Directory for output files
            
        Returns:
            str: Path to the output file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the base filename without extension
        base_name = Path(input_path).stem
        
        # Create output path
        output_path = os.path.join(output_dir, f"{base_name}.json")
        
        return output_path 