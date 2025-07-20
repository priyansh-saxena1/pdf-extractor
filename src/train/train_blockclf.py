#!/usr/bin/env python3
"""
MiniLM Block Classifier Training Script

This script trains a MiniLM-based model for classifying text blocks as
Title, H1-H4, or Other, and exports it to ONNX format.
"""

import os
import sys
import argparse
import logging
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

class BlockClassifierModel(nn.Module):
    """
    MiniLM-based block classifier model.
    """
    
    def __init__(self, vocab_size=30000, hidden_size=384, num_layers=4, num_heads=6, 
                 ff_dim=1536, num_classes=6):
        """
        Initialize the block classifier model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            hidden_size (int): Hidden size of the transformer
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
            num_classes (int): Number of output classes
        """
        super(BlockClassifierModel, self).__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.bbox_embedding = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU()
        )
        self.meta_embedding = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU()
        )
        
        # Transformer layers (simplified implementation)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 32 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, bbox_features, meta_features):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs [batch_size, seq_len]
            bbox_features (torch.Tensor): Bounding box features [batch_size, 4]
            meta_features (torch.Tensor): Meta features [batch_size, 6]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Pass through transformer
        transformer_output = self.transformer(token_embeds)  # [batch_size, seq_len, hidden_size]
        
        # Get CLS token output (first token)
        cls_output = transformer_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Process bbox and meta features
        bbox_output = self.bbox_embedding(bbox_features)  # [batch_size, 32]
        meta_output = self.meta_embedding(meta_features)  # [batch_size, 32]
        
        # Concatenate features
        combined = torch.cat([cls_output, bbox_output, meta_output], dim=1)  # [batch_size, hidden_size+64]
        
        # Classification
        logits = self.classifier(combined)  # [batch_size, num_classes]
        
        return logits


class BlockDataset(Dataset):
    """
    Dataset for block classifier training.
    """
    
    def __init__(self, data_file, max_length=64):
        """
        Initialize the dataset.
        
        Args:
            data_file (str): Path to the data file
            max_length (int): Maximum sequence length
        """
        self.data = self.load_data(data_file)
        self.max_length = max_length
        
        # Class mapping
        self.class_map = {
            "Title": 0,
            "H1": 1,
            "H2": 2,
            "H3": 3,
            "H4": 4,
            "Other": 5
        }
        
    def load_data(self, data_file):
        """
        Load data from file.
        
        Args:
            data_file (str): Path to the data file
            
        Returns:
            list: List of data samples
        """
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load data from {data_file}: {e}")
            return []
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (input_ids, bbox_features, meta_features, label)
        """
        sample = self.data[idx]
        
        # Process text
        text = sample.get("text", "")
        tokens = text.lower().split()[:self.max_length]
        
        # Simple tokenization (in a real implementation, use a proper tokenizer)
        input_ids = np.zeros(self.max_length, dtype=np.int64)
        for i, token in enumerate(tokens):
            if i < self.max_length:
                # Simple hash-based token ID generation
                input_ids[i] = hash(token) % 30000
        
        # Process bbox features
        bbox = sample.get("bbox_norm", [0, 0, 1, 1])
        bbox_features = np.array(bbox, dtype=np.float32)
        
        # Process meta features
        meta = [
            sample.get("area_ratio", 0.0),
            sample.get("aspect_ratio", 0.0),
            sample.get("y_center", 0.5),
            sample.get("confidence", 0.0),
            sample.get("font_ratio", 0.0),
            sample.get("whitespace_ratio", 0.0)
        ]
        meta_features = np.array(meta, dtype=np.float32)
        
        # Get label
        label_str = sample.get("level", "Other")
        label = self.class_map.get(label_str, 5)  # Default to "Other"
        
        return input_ids, bbox_features, meta_features, label


class BlockClassifierTrainer:
    """
    Trainer for MiniLM block classifier model.
    """
    
    def __init__(self, train_file, output_dir='models', batch_size=32, epochs=10, lr=5e-5):
        """
        Initialize the block classifier trainer.
        
        Args:
            train_file (str): Path to the training data file
            output_dir (str): Directory to save the trained model
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            lr (float): Learning rate
        """
        self.train_file = train_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self):
        """
        Train the MiniLM block classifier model.
        """
        try:
            # Create dataset and dataloader
            dataset = BlockDataset(self.train_file)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Create model
            model = BlockClassifierModel()
            model.to(self.device)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=self.lr)
            
            # Training loop
            logger.info(f"Starting training for {self.epochs} epochs...")
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for input_ids, bbox_features, meta_features, labels in dataloader:
                    # Move data to device
                    input_ids = input_ids.to(self.device)
                    bbox_features = bbox_features.to(self.device)
                    meta_features = meta_features.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(input_ids, bbox_features, meta_features)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                # Print epoch statistics
                epoch_loss = running_loss / len(dataloader)
                epoch_acc = 100 * correct / total
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
            # Save the model
            model_path = os.path.join(self.output_dir, 'blockclf.pt')
            torch.save(model.state_dict(), model_path)
            
            # Export to ONNX
            self.export_to_onnx(model)
            
            logger.info(f"Training completed successfully.")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)
    
    def export_to_onnx(self, model):
        """
        Export the trained model to ONNX format.
        
        Args:
            model (BlockClassifierModel): Trained model
        """
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy inputs
            dummy_input_ids = torch.zeros(1, 64, dtype=torch.int64).to(self.device)
            dummy_bbox = torch.zeros(1, 4, dtype=torch.float32).to(self.device)
            dummy_meta = torch.zeros(1, 6, dtype=torch.float32).to(self.device)
            
            # Export to ONNX
            onnx_path = os.path.join(self.output_dir, 'blockclf.onnx')
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_bbox, dummy_meta),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'bbox_features', 'meta_features'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'bbox_features': {0: 'batch_size'},
                    'meta_features': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Quantize the model
            self.quantize_onnx(onnx_path)
            
            logger.info(f"Model exported to ONNX format")
            
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
            quant_path = os.path.join(self.output_dir, 'blockclf_quant.onnx')
            
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


def main():
    """
    Main entry point for the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MiniLM block classifier model')
    parser.add_argument('--train-file', required=True, help='Path to training data file')
    parser.add_argument('--output-dir', default='models', help='Output directory for trained model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()
    
    # Create and run the trainer
    trainer = BlockClassifierTrainer(
        train_file=args.train_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
    
    trainer.train()


if __name__ == "__main__":
    main() 