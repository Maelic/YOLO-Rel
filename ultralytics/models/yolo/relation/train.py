# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Relationship Detection Training Pipeline.

This module implements the trainer for relationship detection models.
"""

from copy import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from ultralytics.data import build_yolo_relation_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.models import yolo
from ultralytics.nn.tasks import RelationModel


class RelationTrainer(DetectionTrainer):
    """
    Trainer for relationship detection models.
    
    This trainer extends DetectionTrainer to handle relationship prediction
    alongside object detection, with support for multi-task learning.
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        Initialize relationship trainer.
        
        Args:
            cfg (str, optional): Path to configuration file
            overrides (dict, optional): Configuration overrides
            _callbacks (dict, optional): Training callbacks
        """
        # Use DEFAULT_CFG if cfg is None
        if cfg is None:
            from ultralytics.cfg import DEFAULT_CFG
            cfg = DEFAULT_CFG
        
        # Extract custom relationship arguments before passing to parent
        custom_relation_args = {}
        if overrides:
            relation_keys = [
                'max_relations_per_image', 'relation_classes',
                'relation_file_train', 'relation_file_val', 'relation_file_test'
            ]
            # Extract custom args
            for key in relation_keys:
                if key in overrides:
                    custom_relation_args[key] = overrides.pop(key)
        
        # Initialize parent with only YOLO-recognized arguments
        super().__init__(cfg, overrides, _callbacks)
        
        # Set relationship-specific parameters
        self.max_relations_per_image = custom_relation_args.get('max_relations_per_image', 
                                                              getattr(self.args, 'max_relations_per_image', 100))
        self.relation_classes = custom_relation_args.get('relation_classes', 
                                                        getattr(self.args, 'relation_classes', []))
        self.relation_file_train = custom_relation_args.get('relation_file_train',
                                                           getattr(self.args, 'relation_file_train', None))
        self.relation_file_val = custom_relation_args.get('relation_file_val',
                                                         getattr(self.args, 'relation_file_val', None))
        
        # Loss tracking
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "rel_loss")
        
        if self.relation_file_train:
            LOGGER.info(f"Training with relationships: {self.relation_file_train}")
        if self.relation_file_val:
            LOGGER.info(f"Validating with relationships: {self.relation_file_val}")
    
    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """
        Build relationship dataset for training or validation.
        
        Args:
            img_path (str): Path to images
            mode (str): Dataset mode ('train' or 'val')
            batch (int, optional): Batch size for rect mode
            
        Returns:
            RelationDataset: Configured relationship dataset
        """
        # Get appropriate relation file from dataset config or trainer args
        relation_file = None
        
        # First, try to get from dataset configuration
        if hasattr(self, 'data') and self.data:
            # Try new format: relation_file_train, relation_file_val
            relation_key = f"relation_file_{mode}"
            if relation_key in self.data:
                relation_file = self.data[relation_key]
            else:
                # Fall back to old format: relation_train, relation_val
                relation_key = f"relation_{mode}"
                if relation_key in self.data:
                    relation_file = self.data[relation_key]
        
        # Fall back to trainer-specific relation file arguments
        if not relation_file:
            if mode == "train" and self.relation_file_train:
                relation_file = self.relation_file_train
            elif mode == "val" and self.relation_file_val:
                relation_file = self.relation_file_val
        
        # If we have a relation file, make it absolute path relative to dataset path
        if relation_file:
            import os
            from pathlib import Path
            
            # If relation_file is relative, make it relative to dataset path
            if not os.path.isabs(relation_file) and hasattr(self, 'data') and 'path' in self.data:
                relation_file = str(Path(self.data['path']) / relation_file)
            
            LOGGER.info(f"Using relation file for {mode}: {relation_file}")
        
        if relation_file:
            # Build relationship dataset
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return build_yolo_relation_dataset(
                self.args,
                img_path=img_path,
                relation_file=relation_file,
                mode=mode,
                stride=gs,
                batch=batch or self.args.batch_size,
                data=self.data,
            )
        else:
            # Fall back to standard detection dataset
            LOGGER.warning(f"No relation file provided for {mode} mode, using standard detection dataset")
            return super().build_dataset(img_path, mode, batch)
    
    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Get relationship detection model.
        
        Args:
            cfg (str, optional): Model configuration path
            weights (str, optional): Model weights path
            verbose (bool): Whether to print model info
            
        Returns:
            RelationModel: Configured relationship model
        """
        # Get relation classes from dataset configuration
        relation_classes = {}
        if hasattr(self, 'data') and self.data:
            # Try to get relation classes from data config
            relation_classes = self.data.get('rel_names', {})
        
        # Fall back to args if not found in data
        if not relation_classes:
            relation_classes = getattr(self.args, 'rel_names', {})
        
        # Log the number of relation classes found
        if relation_classes:
            LOGGER.info(f"Found {len(relation_classes)} relation classes in dataset configuration")
        else:
            LOGGER.warning("No relation classes found in dataset configuration or arguments")
        
        # Create relationship model
        model = RelationModel(
            cfg=cfg or self.args.model,
            ch=self.data.get("channels", 3) if hasattr(self, 'data') and self.data else 3,
            nc=self.data["nc"] if hasattr(self, 'data') and self.data else 80,
            relation_classes=relation_classes,
            verbose=verbose and RANK == -1
        )
        
        if weights:
            # Check if weights are from a detection model (like yolo11n.pt)
            if str(weights).endswith('.pt') and not 'rel' in str(weights).lower():
                # Use our custom detection weight transfer
                LOGGER.info(f"Loading detection weights from {weights} using custom transfer")
                stats = model.load_detection_weights(weights, verbose=verbose)
                if verbose:
                    LOGGER.info(f"Transferred {stats['total_loaded']} weights from detection model")
            else:
                # Standard weight loading for relation model weights
                model.load(weights)
        
        return model
    
    def preprocess_batch(self, batch: Dict) -> Dict:
        """
        Preprocess batch for relationship training.
        
        Args:
            batch (Dict): Training batch
            
        Returns:
            Dict: Preprocessed batch
        """
        # Standard preprocessing
        batch = super().preprocess_batch(batch)
        
        # Move relationship data to device if present
        if "relation_labels" in batch:
            batch["relation_labels"] = batch["relation_labels"].to(self.device, non_blocking=True)
        if "object_pairs" in batch:
            batch["object_pairs"] = batch["object_pairs"].to(self.device, non_blocking=True)
        
        return batch
    
    def get_validator(self):
        """Return a RelationValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "rel_loss"
        return yolo.relation.RelationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def label_loss_items(self, loss_items: Optional[torch.Tensor] = None, prefix: str = "train") -> Dict:
        """
        Returns a loss dict with labeled training loss items.
        
        Args:
            loss_items (torch.Tensor, optional): Loss items tensor
            prefix (str): Prefix for loss names
            
        Returns:
            Dict: Labeled loss dictionary
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys
    
    def progress_string(self) -> str:
        """Return a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size"
        )
    
    def set_model_attributes(self):
        """Set model attributes for relationship training."""
        super().set_model_attributes()
        
        # Set relationship-specific attributes
        if hasattr(self.model, 'relation_classes'):
            LOGGER.info(f"Model initialized with {len(self.model.relation_classes)} relationship classes")



