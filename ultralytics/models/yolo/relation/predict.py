# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Relationship Detection Prediction.
This module implements prediction for relationship detection models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import LOGGER, ops


class RelationPredictor(DetectionPredictor):
    """
    Predictor for relationship detection models.
    
    This class extends DetectionPredictor to handle relationship prediction
    alongside object detection.
    
    Attributes:
        relation_names (dict): Dictionary mapping relation class IDs to names
        relation_threshold (float): Confidence threshold for relationship predictions
        
    Examples:
        >>> predictor = RelationPredictor(cfg=cfg)
        >>> results = predictor(source="image.jpg")
        >>> for result in results:
        ...     print(f"Detected {len(result.relations)} relationships")
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        Initialize RelationPredictor.
        
        Args:
            cfg (str, optional): Path to configuration file
            overrides (dict, optional): Configuration overrides
            _callbacks (dict, optional): Prediction callbacks
        """
        super().__init__(cfg, overrides, _callbacks)
        
        # Relationship-specific settings
        self.relation_names = getattr(self.args, 'relation_names', {})
        self.relation_threshold = getattr(self.args, 'relation_conf', 0.5)
        
        LOGGER.info(f"RelationPredictor initialized with {len(self.relation_names)} relation classes")

    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions to return Results objects."""
        # Handle different prediction formats
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            detection_preds, relation_preds = preds
            
            # Post-process detection predictions
            preds = ops.non_max_suppression(
                detection_preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
            )
            
            # Create Results objects with both detection and relation data
            results = []
            for i, (pred, relation_pred) in enumerate(zip(preds, relation_preds)):
                orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
                img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
                
                # Process relationships
                relations = self._process_relations(pred, relation_pred)
                
                # Create Results object
                results.append(Results(
                    orig_img=orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6] if len(pred) else torch.zeros(0, 6),
                    relations=relations
                ))
            
            return results
        else:
            # Standard detection only
            return super().postprocess(preds, img, orig_imgs)

    def _process_relations(self, detection_pred, relation_pred):
        """Process relationship predictions."""
        if not isinstance(relation_pred, torch.Tensor) or len(relation_pred) == 0:
            return []
        
        # Extract relationship data
        # Assuming relation_pred format: [obj1_idx, obj2_idx, rel_class, confidence]
        relations = []
        
        for rel in relation_pred:
            if len(rel) >= 4:
                obj1_idx, obj2_idx, rel_class, confidence = rel[:4]
                
                # Filter by confidence threshold
                if confidence >= self.relation_threshold:
                    relation_name = self.relation_names.get(int(rel_class), f'relation_{int(rel_class)}')
                    
                    relations.append({
                        'object1_idx': int(obj1_idx),
                        'object2_idx': int(obj2_idx),
                        'relation_class': int(rel_class),
                        'relation_name': relation_name,
                        'confidence': float(confidence)
                    })
        
        return relations

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        # Call parent method for detection results
        super().write_results(idx, results, batch)
        
        # Write relationship-specific results
        for result in results:
            if hasattr(result, 'relations') and result.relations:
                self._write_relation_results(result)

    def _write_relation_results(self, result):
        """Write relationship detection results to file."""
        if not hasattr(result, 'relations') or not result.relations:
            return
        
        # Create relation results file
        save_path = self.save_dir / 'relations' / f'{result.path.stem}_relations.txt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            for relation in result.relations:
                f.write(f"{relation['object1_idx']} {relation['object2_idx']} "
                       f"{relation['relation_class']} {relation['confidence']:.6f}\n")
        
        LOGGER.info(f"Relationship results saved to {save_path}")
