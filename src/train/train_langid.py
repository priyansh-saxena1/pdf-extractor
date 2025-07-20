#!/usr/bin/env python3
"""
FastText Language Identification Training Script

This script trains a FastText model for language/script identification
and exports it to ONNX format.
"""

import os
import sys
import argparse
import logging
import numpy as np
import fasttext
import onnx
import onnxruntime
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

class LanguageIdTrainer:
    """
    Trainer for FastText language identification model.
    """
    
    def __init__(self, train_file, output_dir='models', dim=300, epoch=25, lr=0.1):
        """
        Initialize the language identification trainer.
        
        Args:
            train_file (str): Path to the training data file
            output_dir (str): Directory to save the trained model
            dim (int): Embedding dimension
            epoch (int): Number of training epochs
            lr (float): Learning rate
        """
        self.train_file = train_file
        self.output_dir = output_dir
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self):
        """
        Train the FastText language identification model.
        """
        try:
            # Train FastText model
            logger.info(f"Training FastText model with {self.epoch} epochs...")
            model = fasttext.train_supervised(
                input=self.train_file,
                lr=self.lr,
                dim=self.dim,
                epoch=self.epoch,
                wordNgrams=2,
                minn=3,
                maxn=6
            )
            
            # Save the model
            model_path = os.path.join(self.output_dir, 'langid.bin')
            model.save_model(model_path)
            
            # Evaluate the model
            self.evaluate_model(model)
            
            # Export to ONNX
            self.export_to_onnx(model)
            
            logger.info(f"Training completed successfully.")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)
    
    def evaluate_model(self, model):
        """
        Evaluate the trained model.
        
        Args:
            model (fasttext.FastText): Trained FastText model
        """
        try:
            # Evaluate on test set if available
            test_file = self.train_file.replace('train', 'test')
            if os.path.exists(test_file):
                result = model.test(test_file)
                logger.info(f"Test results: samples={result[0]}, precision={result[1]}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    
    def export_to_onnx(self, model):
        """
        Export the trained model to ONNX format with quantization.
        
        Args:
            model (fasttext.FastText): Trained FastText model
        """
        try:
            import torch
            import torch.nn as nn
            
            # Create a PyTorch wrapper for FastText
            class FastTextONNX(nn.Module):
                def __init__(self, ft_model):
                    super(FastTextONNX, self).__init__()
                    self.ft_model = ft_model
                    
                    # Extract weights from FastText model
                    self.embedding_dim = ft_model.get_dimension()
                    self.num_classes = len(ft_model.get_labels())
                    
                    # Create embedding layer
                    self.embedding = nn.Embedding(30000, self.embedding_dim)  # Simplified vocab size
                    
                    # Create output layer
                    self.fc = nn.Linear(self.embedding_dim, self.num_classes)
                    
                    # Initialize with FastText weights (simplified)
                    # In a real implementation, you would extract the actual weights
                    
                def forward(self, x):
                    # x is byte sequence
                    # Convert to token IDs (simplified)
                    token_ids = x.long() % 30000
                    
                    # Get embeddings
                    embeds = self.embedding(token_ids)
                    
                    # Average embeddings
                    avg_embed = torch.mean(embeds, dim=1)
                    
                    # Output layer
                    logits = self.fc(avg_embed)
                    
                    return torch.softmax(logits, dim=1)
            
            # Create PyTorch model
            pt_model = FastTextONNX(model)
            pt_model.eval()
            
            # Create dummy input
            dummy_input = torch.zeros(1, 300, dtype=torch.float32)
            
            # Export to ONNX
            onnx_path = os.path.join(self.output_dir, 'langid.onnx')
            torch.onnx.export(
                pt_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                             'output': {0: 'batch_size'}}
            )
            
            # Quantize the model
            self.quantize_onnx(onnx_path)
            
            logger.info(f"Model exported to ONNX format")
            
        except ImportError:
            logger.error("PyTorch is required for ONNX export. Please install it first.")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    
    def quantize_onnx(self, onnx_path):
        """
        Quantize the ONNX model to INT8.
        
        Args:
            onnx_path (str): Path to the ONNX model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Output path for quantized model
            quant_path = os.path.join(self.output_dir, 'langid_quant.onnx')
            
            # Quantize the model
            quantize_dynamic(
                model_input=onnx_path,
                model_output=quant_path,
                weight_type=QuantType.QInt8
            )
            
            logger.info(f"Model quantized to {quant_path}")
            
        except ImportError:
            logger.error("onnxruntime-tools is required for quantization. Please install it first.")
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
    
    def prepare_training_data(self, text_files, output_file):
        """
        Prepare training data for FastText from text files.
        
        Args:
            text_files (list): List of text files with language labels
            output_file (str): Output file path for FastText training
            
        Returns:
            str: Path to the prepared training file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for text_file in text_files:
                    # Extract language from filename or path
                    lang = os.path.basename(text_file).split('.')[0]
                    
                    # Read text file
                    with open(text_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                        for line in f_in:
                            line = line.strip()
                            if line:
                                # Format for FastText: __label__LANG text
                                f_out.write(f"__label__{lang} {line}\n")
                                
            logger.info(f"Training data prepared at {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return None


def main():
    """
    Main entry point for the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train FastText language identification model')
    parser.add_argument('--train-file', required=True, help='Path to training data file')
    parser.add_argument('--output-dir', default='models', help='Output directory for trained model')
    parser.add_argument('--dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--epoch', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()
    
    # Create and run the trainer
    trainer = LanguageIdTrainer(
        train_file=args.train_file,
        output_dir=args.output_dir,
        dim=args.dim,
        epoch=args.epoch,
        lr=args.lr
    )
    
    trainer.train()


if __name__ == "__main__":
    main() 