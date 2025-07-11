# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Relationship Detection Validation.
This module implements validation for relationship detection models.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou, RelMetrics


class RelationValidator(DetectionValidator):
    """
    Validator for relationship detection models.
    
    This class extends DetectionValidator to handle relationship prediction
    validation alongside object detection.
    
    Attributes:
        relationship_metrics (dict): Metrics for relationship prediction
        relation_confusion_matrix (ConfusionMatrix): Confusion matrix for relationships
        relation_stats (list): Statistics for relationship predictions
        
    Examples:
        >>> validator = RelationValidator(args=args)
        >>> metrics = validator(model=model)
        >>> print(f"Relationship mAP@0.5: {metrics.relation_map50}")
    """
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize RelationValidator.
        
        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader for validation dataset
            save_dir (Path, optional): Directory to save validation results
            pbar (tqdm, optional): Progress bar for displaying validation progress
            args (SimpleNamespace, optional): Configuration arguments
            _callbacks (dict, optional): Dictionary of callback functions
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        
        # Relationship-specific attributes
        self.relation_nc = getattr(args, 'relation_nc', 56)  # Default to PSG relation classes
        self.relation_names = getattr(args, 'relation_names', {i: f'relation_{i}' for i in range(self.relation_nc)})
        
        # Use RelMetrics for comprehensive relation evaluation
        self.relation_metrics = None
        
        # Storage for relation predictions and ground truth
        self.relation_preds = []  # [(subj_idx, obj_idx, rel_cls, conf), ...]
        self.relation_gts = []    # [(subj_idx, obj_idx, rel_cls), ...]
        
        LOGGER.info(f"RelationValidator initialized with {self.relation_nc} relation classes")

    def init_metrics(self, model):
        """Initialize metrics for both detection and relationship validation."""
        super().init_metrics(model)
        
        # Initialize relationship-specific metrics using RelMetrics
        self.relation_metrics = RelMetrics(save_dir=self.save_dir, plot=self.args.plots, names=self.relation_names)
        
        # Check if model has relationship components
        if hasattr(model, 'relationship_head'):
            LOGGER.info("✅ Model has relationship head - relationship validation enabled")
        else:
            LOGGER.warning("⚠️  Model missing relationship head - only detection will be validated")

    def preprocess(self, batch):
        """Preprocess batch for validation including relationship data."""
        batch = super().preprocess(batch)
        
        # Handle relationship data if present
        if 'relations' in batch:
            batch['relations'] = [rel for rel in batch['relations']]
        if 'relation_labels' in batch:
            batch['relation_labels'] = batch['relation_labels'].to(self.device, non_blocking=True)
        if 'object_pairs' in batch:
            batch['object_pairs'] = [pairs for pairs in batch['object_pairs']]
        if 'num_relations' in batch:
            batch['num_relations'] = batch['num_relations'].to(self.device, non_blocking=True)
            
        return batch

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        # Handle both detection and relationship predictions
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            detection_preds, relation_preds = preds
            detection_preds = ops.non_max_suppression(
                detection_preds,
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls,
                max_det=self.args.max_det,
            )
            return detection_preds, relation_preds
        else:
            # Standard detection only
            return super().postprocess(preds)

    def update_metrics(self, preds, batch):
        """Update metrics for both detection and relationships."""
        # Handle different prediction formats
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            detection_preds, relation_preds = preds
            
            # Update detection metrics
            super().update_metrics(detection_preds, batch)
            
            # Update relationship metrics
            self._update_relationship_metrics(detection_preds, relation_preds, batch)
        else:
            # Standard detection only
            super().update_metrics(preds, batch)

    def _update_relationship_metrics(self, detection_preds, relation_preds, batch):
        """Update relationship-specific metrics."""
        if not hasattr(batch, 'relations') and 'relations' not in batch:
            return
            
        for si, (det_pred, rel_pred) in enumerate(zip(detection_preds, relation_preds)):
            # Get ground truth relationships for this sample
            if 'relations' in batch and si < len(batch['relations']):
                target_relations = batch['relations'][si]
            else:
                target_relations = np.array([])
                
            if len(target_relations) == 0:
                continue
            
            # Get object pairs from ground truth relations
            # target_relations format: [subj_idx, obj_idx, rel_cls]
            gt_pairs = target_relations[:, :2].astype(int)  # Object pairs
            gt_rel_cls = target_relations[:, 2].astype(int)  # Relation classes
            
            # Process relationship predictions
            if isinstance(rel_pred, torch.Tensor) and len(rel_pred) > 0:
                # Extract relationship predictions
                rel_conf = rel_pred[:, -1]  # Confidence scores
                rel_cls = rel_pred[:, -2].int()  # Relationship classes
                
                # Get predicted object pairs (assuming they're in rel_pred[:, :2])
                pred_pairs = rel_pred[:, :2].int()  # Object pairs indices
                
                # Convert to format for relation metrics: [(subj_idx, obj_idx, rel_cls, conf), ...]
                batch_pred_relations = []
                for i in range(len(rel_pred)):
                    subj_idx, obj_idx = pred_pairs[i].tolist()
                    rel_class = rel_cls[i].item()
                    confidence = rel_conf[i].item()
                    batch_pred_relations.append((subj_idx, obj_idx, rel_class, confidence))
                
                # Convert ground truth to format: [(subj_idx, obj_idx, rel_cls), ...]
                batch_gt_relations = []
                for i in range(len(target_relations)):
                    subj_idx, obj_idx = gt_pairs[i].tolist()
                    rel_class = gt_rel_cls[i]
                    batch_gt_relations.append((subj_idx, obj_idx, rel_class))
                
                # Store for final processing
                self.relation_preds.extend(batch_pred_relations)
                self.relation_gts.extend(batch_gt_relations)

    def finalize_metrics(self, *args, **kwargs):
        """Finalize and compute all metrics."""
        # Finalize detection metrics
        super().finalize_metrics(*args, **kwargs)
        
        # Finalize relationship metrics if we have relation data
        if self.relation_preds and self.relation_gts:
            # Process relation stats with RelMetrics
            self.relation_metrics.process_relation_stats(
                pred_relations=self.relation_preds,
                gt_relations=self.relation_gts,
                relation_names=self.relation_names
            )
            
            # Update main metrics with relationship results
            relation_results = self.relation_metrics.mean_results()
            
            # Extract specific metrics (detection + relation)
            if len(relation_results) >= 10:  # 4 detection + 6 relation metrics
                self.metrics.relation_recall_20 = relation_results[4]   # Recall@20
                self.metrics.relation_mean_recall_20 = relation_results[5]  # MeanRecall@20
                self.metrics.relation_recall_50 = relation_results[6]   # Recall@50
                self.metrics.relation_mean_recall_50 = relation_results[7]  # MeanRecall@50
                self.metrics.relation_recall_100 = relation_results[8]  # Recall@100
                self.metrics.relation_mean_recall_100 = relation_results[9] # MeanRecall@100
            
            LOGGER.info(f"Relationship Validation Results:")
            LOGGER.info(f"  Total predicted relations: {len(self.relation_preds)}")
            LOGGER.info(f"  Total ground truth relations: {len(self.relation_gts)}")
            LOGGER.info(f"  Recall@20: {getattr(self.metrics, 'relation_recall_20', 0.0):.4f}")
            LOGGER.info(f"  MeanRecall@20: {getattr(self.metrics, 'relation_mean_recall_20', 0.0):.4f}")
            LOGGER.info(f"  Recall@50: {getattr(self.metrics, 'relation_recall_50', 0.0):.4f}")
            LOGGER.info(f"  MeanRecall@50: {getattr(self.metrics, 'relation_mean_recall_50', 0.0):.4f}")
            LOGGER.info(f"  Recall@100: {getattr(self.metrics, 'relation_recall_100', 0.0):.4f}")
            LOGGER.info(f"  MeanRecall@100: {getattr(self.metrics, 'relation_mean_recall_100', 0.0):.4f}")
        else:
            LOGGER.info("No relationship predictions or ground truth available for evaluation")
            LOGGER.info(f"  Recall@20: {getattr(self.metrics, 'relation_recall_20', 0.0):.4f}")
            LOGGER.info(f"  MeanRecall@20: {getattr(self.metrics, 'relation_mean_recall_20', 0.0):.4f}")
            LOGGER.info(f"  Recall@50: {getattr(self.metrics, 'relation_recall_50', 0.0):.4f}")
            LOGGER.info(f"  MeanRecall@50: {getattr(self.metrics, 'relation_mean_recall_50', 0.0):.4f}")
            LOGGER.info(f"  Recall@100: {getattr(self.metrics, 'relation_recall_100', 0.0):.4f}")
            LOGGER.info(f"  MeanRecall@100: {getattr(self.metrics, 'relation_mean_recall_100', 0.0):.4f}")

    def print_results(self):
        """Print validation results for both detection and relationships."""
        super().print_results()
        
        # Print relationship results
        if hasattr(self.metrics, 'relation_recall_20'):
            LOGGER.info(f"\nRelationship Results:")
            LOGGER.info(f"{'Metric':<20} {'@20':<10} {'@50':<10} {'@100':<10}")
            LOGGER.info("-" * 50)
            
            # Print Recall@K
            recall_20 = getattr(self.metrics, 'relation_recall_20', 0.0)
            recall_50 = getattr(self.metrics, 'relation_recall_50', 0.0)
            recall_100 = getattr(self.metrics, 'relation_recall_100', 0.0)
            LOGGER.info(f"{'Recall':<20} {recall_20:<10.3f} {recall_50:<10.3f} {recall_100:<10.3f}")
            
            # Print MeanRecall@K
            mean_recall_20 = getattr(self.metrics, 'relation_mean_recall_20', 0.0)
            mean_recall_50 = getattr(self.metrics, 'relation_mean_recall_50', 0.0)
            mean_recall_100 = getattr(self.metrics, 'relation_mean_recall_100', 0.0)
            LOGGER.info(f"{'MeanRecall':<20} {mean_recall_20:<10.3f} {mean_recall_50:<10.3f} {mean_recall_100:<10.3f}")
            
            LOGGER.info("-" * 50)
            
            # Show fitness score
            if hasattr(self.relation_metrics, 'fitness'):
                LOGGER.info(f"Combined Fitness: {self.relation_metrics.fitness:.3f}")
        else:
            LOGGER.info("No relationship metrics available")

    def save_json(self, stats, save_dir=Path("."), file_name="relation_results.json"):
        """Save relationship validation results to JSON."""
        # Save detection results
        super().save_json(stats)
        
        # Save relationship results
        if hasattr(self.metrics, 'relation_recall_20'):
            relation_results = {
                'recall_at_20': float(getattr(self.metrics, 'relation_recall_20', 0.0)),
                'mean_recall_at_20': float(getattr(self.metrics, 'relation_mean_recall_20', 0.0)),
                'recall_at_50': float(getattr(self.metrics, 'relation_recall_50', 0.0)),
                'mean_recall_at_50': float(getattr(self.metrics, 'relation_mean_recall_50', 0.0)),
                'recall_at_100': float(getattr(self.metrics, 'relation_recall_100', 0.0)),
                'mean_recall_at_100': float(getattr(self.metrics, 'relation_mean_recall_100', 0.0)),
                'relation_classes': self.relation_names,
                'validation_summary': {
                    'total_images': len(self.dataloader.dataset) if self.dataloader else 0,
                    'relation_classes': self.relation_nc,
                    'total_predictions': len(self.relation_preds),
                    'total_ground_truth': len(self.relation_gts),
                }
            }
            
            # Add fitness score if available
            if hasattr(self.relation_metrics, 'fitness'):
                relation_results['fitness'] = float(self.relation_metrics.fitness)
            
            # Save to file
            save_path = save_dir / file_name
            with open(save_path, 'w') as f:
                json.dump(relation_results, f, indent=2)
            
            LOGGER.info(f"Relationship validation results saved to {save_path}")
        else:
            LOGGER.info("No relationship metrics to save")
