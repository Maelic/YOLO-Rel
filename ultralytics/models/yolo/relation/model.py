# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Relationship Detection Model based on YOLOE architecture.
This module implements the core relationship prediction model for open-vocabulary settings.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

from ultralytics.nn.tasks import YOLOEModel
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.utils import LOGGER


class SpatialRelationEncoder(nn.Module):
    """
    Encodes spatial relationships between object pairs using geometric features.
    
    This module computes spatial features from bounding box pairs and encodes them
    into a fixed-size representation suitable for relationship prediction.
    """
    
    def __init__(self, feat_dim: int = 512):
        """
        Initialize the spatial relation encoder.
        
        Args:
            feat_dim (int): Output feature dimension
        """
        super().__init__()
        self.feat_dim = feat_dim
        
        # Spatial feature MLP
        # Input: 11 geometric features per object pair
        # [dx, dy, dw, dh, sx, sy, sw, sh, area_ratio, overlap_ratio, distance]
        self.spatial_mlp = nn.Sequential(
            nn.Linear(11, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_spatial_features(self, bbox_pairs: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial features from bounding box pairs.
        
        Args:
            bbox_pairs (torch.Tensor): Pairs of bounding boxes [N, 2, 4] (x1, y1, x2, y2)
            
        Returns:
            torch.Tensor: Spatial features [N, 11]
        """
        batch_size = bbox_pairs.size(0)
        device = bbox_pairs.device
        
        # Extract subject and object boxes
        subj_boxes = bbox_pairs[:, 0]  # [N, 4]
        obj_boxes = bbox_pairs[:, 1]   # [N, 4]
        
        # Ensure valid bboxes (x2 > x1, y2 > y1)
        subj_boxes = self._ensure_valid_bbox(subj_boxes)
        obj_boxes = self._ensure_valid_bbox(obj_boxes)
        
        # Compute box centers and dimensions
        subj_cx = (subj_boxes[:, 0] + subj_boxes[:, 2]) / 2
        subj_cy = (subj_boxes[:, 1] + subj_boxes[:, 3]) / 2
        subj_w = torch.clamp(subj_boxes[:, 2] - subj_boxes[:, 0], min=1e-6)
        subj_h = torch.clamp(subj_boxes[:, 3] - subj_boxes[:, 1], min=1e-6)
        
        obj_cx = (obj_boxes[:, 0] + obj_boxes[:, 2]) / 2
        obj_cy = (obj_boxes[:, 1] + obj_boxes[:, 3]) / 2
        obj_w = torch.clamp(obj_boxes[:, 2] - obj_boxes[:, 0], min=1e-6)
        obj_h = torch.clamp(obj_boxes[:, 3] - obj_boxes[:, 1], min=1e-6)
        
        # Relative position features (normalized by subject box size)
        dx = (obj_cx - subj_cx) / (subj_w + 1e-6)
        dy = (obj_cy - subj_cy) / (subj_h + 1e-6)
        dw = torch.log(torch.clamp(obj_w / (subj_w + 1e-8), min=1e-6, max=1e6))
        dh = torch.log(torch.clamp(obj_h / (subj_h + 1e-8), min=1e-6, max=1e6))
        
        # Absolute position features (normalized by image size, assuming 640x640)
        sx = torch.clamp(subj_cx / 640.0, 0.0, 1.0)
        sy = torch.clamp(subj_cy / 640.0, 0.0, 1.0)
        sw = torch.clamp(subj_w / 640.0, 0.0, 1.0)
        sh = torch.clamp(subj_h / 640.0, 0.0, 1.0)
        
        # Compute area ratio with better numerical stability
        subj_area = subj_w * subj_h
        obj_area = obj_w * obj_h
        area_ratio = torch.clamp(obj_area / (subj_area + 1e-6), min=1e-6, max=1e6)
        
        # Compute overlap ratio (IoU) with better numerical stability
        overlap_area = self._compute_overlap_area(subj_boxes, obj_boxes)
        union_area = subj_area + obj_area - overlap_area + 1e-6
        overlap_ratio = torch.clamp(overlap_area / union_area, min=0.0, max=1.0)
        
        # Compute distance with numerical stability
        distance = torch.clamp(torch.sqrt((dx * subj_w)**2 + (dy * subj_h)**2 + 1e-8) / 640.0, min=0.0, max=10.0)
        
        # Stack all features
        spatial_features = torch.stack([
            dx, dy, dw, dh, sx, sy, sw, sh, area_ratio, overlap_ratio, distance
        ], dim=1)
        
        # Check for NaN or infinite values and replace them
        spatial_features = torch.where(
            torch.isnan(spatial_features) | torch.isinf(spatial_features),
            torch.zeros_like(spatial_features),
            spatial_features
        )
        
        # Clamp final features to reasonable range
        spatial_features = torch.clamp(spatial_features, min=-10.0, max=10.0)
        
        return spatial_features
    
    def _compute_overlap_area(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute overlap area between two sets of boxes."""
        x1_max = torch.maximum(box1[:, 0], box2[:, 0])
        y1_max = torch.maximum(box1[:, 1], box2[:, 1])
        x2_min = torch.minimum(box1[:, 2], box2[:, 2])
        y2_min = torch.minimum(box1[:, 3], box2[:, 3])
        
        overlap_w = torch.clamp(x2_min - x1_max, min=0)
        overlap_h = torch.clamp(y2_min - y1_max, min=0)
        
        return overlap_w * overlap_h
    
    def _ensure_valid_bbox(self, boxes: torch.Tensor) -> torch.Tensor:
        """Ensure bboxes have x2 > x1 and y2 > y1."""
        # Clamp to ensure x2 > x1 and y2 > y1
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Ensure x2 > x1 and y2 > y1
        x2 = torch.maximum(x2, x1 + 1e-6)
        y2 = torch.maximum(y2, y1 + 1e-6)
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def forward(self, bbox_pairs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial encoder.
        
        Args:
            bbox_pairs (torch.Tensor): Pairs of bounding boxes [N, 2, 4]
            
        Returns:
            torch.Tensor: Encoded spatial features [N, feat_dim]
        """
        spatial_feats = self.compute_spatial_features(bbox_pairs)
        return self.spatial_mlp(spatial_feats)


class RelationshipHead(nn.Module):
    """
    Multi-modal relationship prediction head.
    
    This head combines visual features, spatial features, and optional text features
    to predict relationships between object pairs.
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        spatial_dim: int = 512,
        text_dim: int = 512,
        hidden_dim: int = 1024,
        num_relations: int = 50,
        dropout: float = 0.1
    ):
        """
        Initialize relationship head.
        
        Args:
            visual_dim (int): Visual feature dimension
            spatial_dim (int): Spatial feature dimension  
            text_dim (int): Text feature dimension
            hidden_dim (int): Hidden layer dimension
            num_relations (int): Number of relationship classes
            dropout (float): Dropout rate
        """
        super().__init__()
        self.visual_dim = visual_dim
        self.spatial_dim = spatial_dim
        self.text_dim = text_dim
        self.num_relations = num_relations
        
        # Visual feature fusion for subject-object pairs
        self.visual_fusion = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Spatial feature projection
        self.spatial_proj = nn.Linear(spatial_dim, visual_dim)
        
        # Text feature projection (optional)
        self.text_proj = nn.Linear(text_dim, visual_dim)
        
        # Final relationship classifier
        self.relation_classifier = nn.Sequential(
            nn.Linear(visual_dim * 3, hidden_dim),  # visual_subj + visual_obj + spatial
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_relations)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        spatial_features: torch.Tensor,
        object_pairs: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through relationship head.
        
        Args:
            visual_features (torch.Tensor): Visual features from objects [B, N, visual_dim]
            spatial_features (torch.Tensor): Spatial features [B, M, spatial_dim]
            object_pairs (torch.Tensor): Object pair indices [B, M, 2]
            text_features (torch.Tensor, optional): Text features [B, M, text_dim]
            
        Returns:
            torch.Tensor: Relationship predictions [B, M, num_relations]
        """
        batch_size, num_pairs = object_pairs.size(0), object_pairs.size(1)
        num_objects = visual_features.size(1)
        
        # Ensure all indices are within bounds
        subj_indices = torch.clamp(object_pairs[:, :, 0], 0, num_objects - 1)  # [B, M]
        obj_indices = torch.clamp(object_pairs[:, :, 1], 0, num_objects - 1)   # [B, M]
        
        # Gather visual features for subjects and objects
        subj_visual = torch.gather(
            visual_features, 1, 
            subj_indices.unsqueeze(-1).expand(-1, -1, self.visual_dim)
        )  # [B, M, visual_dim]
        
        obj_visual = torch.gather(
            visual_features, 1,
            obj_indices.unsqueeze(-1).expand(-1, -1, self.visual_dim)
        )  # [B, M, visual_dim]
        
        # Apply visual attention between subject and object
        pairs_visual = torch.stack([subj_visual, obj_visual], dim=2)  # [B, M, 2, visual_dim]
        pairs_visual = pairs_visual.view(batch_size * num_pairs, 2, self.visual_dim)
        
        attended_visual, _ = self.visual_fusion(
            pairs_visual, pairs_visual, pairs_visual
        )  # [B*M, 2, visual_dim]
        
        attended_visual = attended_visual.view(batch_size, num_pairs, 2, self.visual_dim)
        subj_visual_att = attended_visual[:, :, 0]  # [B, M, visual_dim]
        obj_visual_att = attended_visual[:, :, 1]   # [B, M, visual_dim]
        
        # Project spatial features with NaN checking
        spatial_proj = self.spatial_proj(spatial_features)  # [B, M, visual_dim]
        
        # Check for NaN values in spatial projection
        if torch.isnan(spatial_proj).any():
            spatial_proj = torch.where(torch.isnan(spatial_proj), torch.zeros_like(spatial_proj), spatial_proj)
        
        # Combine all features
        combined_features = torch.cat([
            subj_visual_att, obj_visual_att, spatial_proj
        ], dim=-1)  # [B, M, visual_dim * 3]
        
        # Optional text feature integration
        if text_features is not None:
            text_proj = self.text_proj(text_features)  # [B, M, visual_dim]
            combined_features = torch.cat([combined_features, text_proj], dim=-1)
        
        # Check for NaN values before final prediction
        if torch.isnan(combined_features).any():
            combined_features = torch.where(torch.isnan(combined_features), torch.zeros_like(combined_features), combined_features)
        
        # Predict relationships
        relation_logits = self.relation_classifier(combined_features)  # [B, M, num_relations]
        
        # Final NaN check and clipping
        if torch.isnan(relation_logits).any() or torch.isinf(relation_logits).any():
            relation_logits = torch.where(
                torch.isnan(relation_logits) | torch.isinf(relation_logits),
                torch.zeros_like(relation_logits),
                relation_logits
            )
        
        # Clip logits to prevent extreme values
        relation_logits = torch.clamp(relation_logits, min=-10.0, max=10.0)
        
        return relation_logits


class RelationModel(YOLOEModel):
    """
    Relationship detection model extending YOLOE for open-vocabulary relationship prediction.
    
    This model combines object detection with relationship prediction, supporting both
    text and visual prompts for open-vocabulary scenarios.
    """
    
    def __init__(
        self,
        cfg: str = "yoloe-v8s.yaml",
        ch: int = 3,
        nc: Optional[int] = None,
        relation_classes = None,
        verbose: bool = True
    ):
        """
        Initialize relationship model.
        
        Args:
            cfg (str): Model configuration file path
            ch (int): Number of input channels
            nc (int, optional): Number of object classes
            relation_classes (List[str] or Dict[int, str], optional): Relationship class names
            verbose (bool): Whether to display model information
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        
        # Handle different relation_classes formats
        if isinstance(relation_classes, dict):
            # Convert dict to list (assumes keys are consecutive integers starting from 0)
            max_key = max(relation_classes.keys()) if relation_classes else -1
            self.relation_classes = [""] * (max_key + 1)
            for k, v in relation_classes.items():
                self.relation_classes[k] = v
        elif isinstance(relation_classes, list):
            self.relation_classes = relation_classes
        elif relation_classes is None:
            # Default relationship classes (expandable)
            self.relation_classes = [
                "on", "in", "under", "above", "beside", "near", "far",
                "holding", "wearing", "riding", "touching", "using",
                "part_of", "made_of", "belongs_to", "similar_to",
                "left_of", "right_of", "behind", "in_front_of"
            ]
        else:
            LOGGER.warning(f"Invalid relation_classes format: {type(relation_classes)}, using default")
            self.relation_classes = [
                "on", "in", "under", "above", "beside", "near", "far",
                "holding", "wearing", "riding", "touching", "using",
                "part_of", "made_of", "belongs_to", "similar_to",
                "left_of", "right_of", "behind", "in_front_of"
            ]
        
        self.num_relations = len(self.relation_classes)
        
        if verbose:
            LOGGER.info(f"RelationModel initialized with {self.num_relations} relation classes")
        
        # Initialize relationship components
        self.spatial_encoder = SpatialRelationEncoder(feat_dim=512)
        self.relationship_head = RelationshipHead(
            visual_dim=512,
            spatial_dim=512,
            text_dim=512,
            num_relations=self.num_relations
        )
        
        # Store relationship embeddings (for open-vocabulary)
        self.relation_embeddings = None
        
        if verbose:
            LOGGER.info(f"Initialized RelationModel with {self.num_relations} relationship classes")
    
    def set_relation_classes(self, relation_classes: List[str], embeddings: Optional[torch.Tensor] = None):
        """
        Set relationship classes for open-vocabulary prediction.
        
        Args:
            relation_classes (List[str]): List of relationship class names
            embeddings (torch.Tensor, optional): Pre-computed relation embeddings
        """
        self.relation_classes = relation_classes
        self.num_relations = len(relation_classes)
        
        # Update relationship head output size
        self.relationship_head.num_relations = self.num_relations
        self.relationship_head.relation_classifier[-1] = nn.Linear(
            self.relationship_head.relation_classifier[-1].in_features,
            self.num_relations
        )
        
        if embeddings is not None:
            self.relation_embeddings = embeddings
        
        LOGGER.info(f"Updated RelationModel with {self.num_relations} relationship classes")
    
    def generate_object_pairs(self, num_objects: int, max_pairs: int = 100) -> torch.Tensor:
        """
        Generate all possible object pairs for relationship prediction.
        
        Args:
            num_objects (int): Number of detected objects
            max_pairs (int): Maximum number of pairs to generate
            
        Returns:
            torch.Tensor: Object pair indices [num_pairs, 2]
        """
        if num_objects < 2:
            return torch.zeros((0, 2), dtype=torch.long)
        
        # Generate all possible pairs (excluding self-pairs)
        pairs = []
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    pairs.append([i, j])
        
        pairs = torch.tensor(pairs, dtype=torch.long)
        
        # Limit number of pairs if too many
        if len(pairs) > max_pairs:
            # Randomly sample pairs or use some heuristic
            indices = torch.randperm(len(pairs))[:max_pairs]
            pairs = pairs[indices]
        
        return pairs
    
    def predict_relationships(
        self,
        visual_features: torch.Tensor,
        bboxes: torch.Tensor,
        object_pairs: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict relationships between detected objects.
        
        Args:
            visual_features (torch.Tensor): Visual features from objects [B, N, D]
            bboxes (torch.Tensor): Bounding boxes [B, N, 4]
            object_pairs (torch.Tensor, optional): Specific object pairs to evaluate
            text_features (torch.Tensor, optional): Text features for relationships
            
        Returns:
            Dict[str, torch.Tensor]: Relationship predictions and metadata
        """
        batch_size, num_objects = visual_features.size(0), visual_features.size(1)
        
        # Generate object pairs if not provided
        if object_pairs is None:
            object_pairs = []
            for b in range(batch_size):
                pairs = self.generate_object_pairs(num_objects)
                object_pairs.append(pairs)
            
            # Pad pairs to same length
            max_pairs = max(len(pairs) for pairs in object_pairs)
            if max_pairs == 0:
                return {
                    "relation_logits": torch.zeros((batch_size, 0, self.num_relations), device=visual_features.device, dtype=visual_features.dtype),
                    "object_pairs": torch.zeros((batch_size, 0, 2), dtype=torch.long, device=visual_features.device),
                    "spatial_features": torch.zeros((batch_size, 0, 512), device=visual_features.device, dtype=visual_features.dtype)
                }
            
            padded_pairs = torch.zeros((batch_size, max_pairs, 2), dtype=torch.long)
            for b, pairs in enumerate(object_pairs):
                if len(pairs) > 0:
                    padded_pairs[b, :len(pairs)] = pairs
            
            object_pairs = padded_pairs
        
        # Extract bounding box pairs
        batch_size, num_pairs = object_pairs.size(0), object_pairs.size(1)
        
        # Check for valid indices before creating bbox_pairs
        valid_mask = (
            (object_pairs[:, :, 0] >= 0) & 
            (object_pairs[:, :, 0] < num_objects) &
            (object_pairs[:, :, 1] >= 0) & 
            (object_pairs[:, :, 1] < num_objects)
        )
        
        # Initialize bbox_pairs with zeros
        bbox_pairs = torch.zeros((batch_size, num_pairs, 2, 4), device=bboxes.device)
        
        # Only process valid pairs
        for b in range(batch_size):
            for p in range(num_pairs):
                if valid_mask[b, p]:
                    obj1_idx = object_pairs[b, p, 0].item()
                    obj2_idx = object_pairs[b, p, 1].item()
                    bbox_pairs[b, p, 0] = bboxes[b, obj1_idx]
                    bbox_pairs[b, p, 1] = bboxes[b, obj2_idx]
        
        # Encode spatial features
        bbox_pairs_flat = bbox_pairs.view(-1, 2, 4)
        spatial_features = self.spatial_encoder(bbox_pairs_flat)
        spatial_features = spatial_features.view(batch_size, num_pairs, -1)
        
        # Create a mask for valid pairs to avoid accessing invalid indices in relationship head
        expanded_valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, self.num_relations)
        
        # Predict relationships only for valid pairs
        relation_logits = torch.zeros((batch_size, num_pairs, self.num_relations), device=visual_features.device, dtype=visual_features.dtype)
        
        if valid_mask.sum() > 0:
            # Create temporary object_pairs with only valid indices for the relationship head
            # We need to ensure all indices are within bounds
            temp_object_pairs = object_pairs.clone()
            temp_object_pairs[~valid_mask] = 0  # Set invalid pairs to index 0 (will be masked out anyway)
            
            # Predict relationships
            temp_logits = self.relationship_head(
                visual_features=visual_features,
                spatial_features=spatial_features,
                object_pairs=temp_object_pairs,
                text_features=text_features
            )
            
            # Ensure dtype compatibility before assignment
            if temp_logits.dtype != relation_logits.dtype:
                temp_logits = temp_logits.to(relation_logits.dtype)
            
            # Only keep logits for valid pairs
            relation_logits[valid_mask] = temp_logits[valid_mask]
        
        return {
            "relation_logits": relation_logits,
            "object_pairs": object_pairs,
            "spatial_features": spatial_features
        }
    
    def forward(self, x, targets=None, *args, **kwargs):
        """
        Forward pass with relationship prediction during training.
        
        Args:
            x (torch.Tensor or dict): Input tensor [B, C, H, W] or batch dict
            targets (dict, optional): Training targets including relationship data
            
        Returns:
            torch.Tensor or tuple: During training returns (loss, loss_items), during inference returns predictions
        """
        # Handle case where input is a batch dict (from trainer)
        if isinstance(x, dict):
            img_tensor = x["img"]
            # If targets is None, use the batch dict as targets during training
            if targets is None and self.training:
                targets = x
        else:
            img_tensor = x
        
        # Get detection output from parent model
        det_output = super().forward(img_tensor, *args, **kwargs)
        
        # During training, also extract features for relationship prediction
        # But skip this during model initialization (when spatial_encoder doesn't exist yet)
        if self.training and hasattr(self, 'spatial_encoder'):
            # Extract features for relationship prediction
            batch_size, channels, height, width = img_tensor.shape
            num_objects_per_image = 20
            
            # Create object-level features from input (simplified approach)
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(img_tensor, (4, 4))
            pooled_features = pooled_features.view(batch_size, channels, -1)
            
            object_features = []
            for i in range(num_objects_per_image):
                if i < 16:
                    obj_feat = pooled_features[:, :, i]
                else:
                    obj_feat = pooled_features[:, :, i % 16] + torch.randn_like(pooled_features[:, :, 0]) * 0.1
                object_features.append(obj_feat)
            
            object_features = torch.stack(object_features, dim=1)
            
            # Project features to expected dimension
            if channels != 512:
                if not hasattr(self, '_feature_projection'):
                    self._feature_projection = nn.Linear(channels, 512).to(object_features.device)
                object_features = self._feature_projection(object_features)
            
            # Predict relationships during training
            relation_predictions = None
            if targets is not None and "object_pairs" in targets:
                # Extract predicted bounding boxes from detection output
                feats = det_output[1] if isinstance(det_output, tuple) else det_output
                pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.model[-1].nc + self.model[-1].reg_max * 4, -1) for xi in feats], 2).split(
                    (self.model[-1].reg_max * 4, self.model[-1].nc), 1
                )
                pred_distri = pred_distri.permute(0, 2, 1).contiguous()
                
                # Decode predicted bounding boxes  
                from ultralytics.utils.tal import make_anchors, dist2bbox
                
                dtype = pred_scores.dtype
                device = object_features.device
                stride = self.model[-1].stride
                imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * stride[0]
                anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
                
                # Use DFL if enabled
                if self.model[-1].reg_max > 1:
                    b, a, c = pred_distri.shape
                    proj = torch.arange(self.model[-1].reg_max, dtype=torch.float, device=device)
                    pred_distri = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_distri.dtype))
                pred_bboxes = dist2bbox(pred_distri, anchor_points, xywh=False)
                
                # Use top predicted bboxes for relationship prediction
                num_objects = object_features.shape[1]
                bboxes = pred_bboxes[:, :num_objects, :].contiguous()
                
                # Ensure bbox dtype matches object_features for consistency
                if bboxes.dtype != object_features.dtype:
                    bboxes = bboxes.to(object_features.dtype)
                
                # Predict relationships
                relation_predictions = self.predict_relationships(
                    visual_features=object_features,
                    bboxes=bboxes,
                    object_pairs=targets["object_pairs"]
                )
            
            # If targets are provided, compute loss and return it directly
            if targets is not None:
                # Combine detection output and relationship predictions
                combined_preds = {
                    "detection": det_output,
                    "object_features": object_features,
                    "relation_predictions": relation_predictions,
                    "feature_map": img_tensor
                }
                
                # Compute combined loss (detection + relationship)
                return self.compute_loss(combined_preds, targets)
            else:
                # Return predictions with object features for loss computation
                return {
                    "detection": det_output,
                    "object_features": object_features,
                    "relation_predictions": relation_predictions,
                    "feature_map": img_tensor
                }
        else:
            # During inference or initialization, return standard detection output
            return det_output
    
    def compute_loss(self, preds, targets):
        """
        Compute combined detection and relationship loss.
        
        Args:
            preds (dict): Model predictions containing detection and relationship features
            targets (dict): Training targets
            
        Returns:
            tuple: (total_loss, loss_items)
        """
        # Initialize the loss criterion if not already done
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
        
        # Compute loss using the relationship loss criterion
        return self.criterion(preds, targets)

    def init_criterion(self):
        """Initialize loss criterion for relationship detection."""
        from ultralytics.utils.loss import RelationLoss
        
        # Ensure model has args attribute for loss initialization
        if not hasattr(self, 'args'):
            from ultralytics.cfg import DEFAULT_CFG
            self.args = DEFAULT_CFG
        
        return RelationLoss(self)
