#!/usr/bin/env python3
"""
Test script for YOLO Relation Detection Training.

This script tests the complete relation detection pipeline using the PSG dataset
and validates the integration of the new 'relation' task into the YOLO framework.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def check_relation_task_integration():
    """Test if the relation task is properly integrated into YOLO."""
    print("🔍 Checking relation task integration...")
    
    # Test YOLO task registration
    try:
        model = YOLO()
        task_map = model.task_map
        
        if 'relation' in task_map:
            print("✅ 'relation' task successfully registered in YOLO")
            relation_components = task_map['relation']
            print(f"  - Model: {relation_components['model']}")
            print(f"  - Trainer: {relation_components['trainer']}")
            print(f"  - Validator: {relation_components['validator']}")
            print(f"  - Predictor: {relation_components['predictor']}")
            return True
        else:
            print("❌ 'relation' task not found in YOLO task_map")
            print(f"Available tasks: {list(task_map.keys())}")
            return False
    except Exception as e:
        print(f"❌ Error checking task integration: {e}")
        return False


def check_relation_components():
    """Test if all relation components can be imported."""
    print("\n🔍 Checking relation component imports...")
    
    try:
        from ultralytics.models.yolo.relation import (
            RelationTrainer, RelationValidator, RelationPredictor
        )
        print("✅ All relation components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error for relation components: {e}")
        return False


def check_psg_config():
    """Check if PSG configuration file exists and is valid."""
    print("\n🔍 Checking PSG configuration...")
    
    psg_config_path = Path("ultralytics/cfg/datasets/PSG.yaml")
    
    if not psg_config_path.exists():
        print(f"❌ PSG config file not found at {psg_config_path}")
        return False
    
    try:
        with open(psg_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'names', 'rel_names']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field '{field}' in PSG config")
                return False
        
        print("✅ PSG configuration file is valid")
        print(f"  - Dataset path: {config['path']}")
        print(f"  - Object classes: {len(config['names'])}")
        print(f"  - Relation classes: {len(config['rel_names'])}")
        
        return True, config
    except Exception as e:
        print(f"❌ Error reading PSG config: {e}")
        return False, None


def test_relation_training(config_path="ultralytics/cfg/datasets/PSG.yaml", 
                          epochs=1, batch_size=2, imgsz=640):
    """Test relation training with minimal setup."""
    print(f"\n🚀 Starting relation training test...")
    print(f"Config: {config_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {imgsz}")
    
    try:
        # Initialize YOLO model with relation task
        print("Initializing YOLO model with relation task...")
        model = YOLO(task='relation')
        
        # Debug: Check if task is properly set
        print(f"Model task: {model.task}")
        print(f"Available tasks: {list(model.task_map.keys())}")
        
        # Check if relation task components are loaded
        if hasattr(model, 'task') and model.task == 'relation':
            print("✅ Relation task properly set")
            trainer_class = model.task_map[model.task]['trainer']
            print(f"Trainer class: {trainer_class}")
        else:
            print(f"❌ Task not set to relation. Current task: {getattr(model, 'task', 'unknown')}")
        
        # Training arguments
        train_args = {
            'data': config_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'project': 'runs/relation',
            'name': 'test_relation_training',
            'save': True,
            'verbose': True,
            'exist_ok': True,
            # Note: Custom relation arguments will be handled by the RelationTrainer
        }
        
        print(f"Training arguments: {train_args}")
        
        # Start training
        print("Starting training...")
        results = model.train(**train_args)
        
        print("✅ Relation training completed successfully!")
        print(f"Training results: {results}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Error during relation training: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_relation_validation(model_path, config_path="ultralytics/cfg/datasets/PSG.yaml"):
    """Test relation validation."""
    print(f"\n🔍 Testing relation validation...")
    
    try:
        # Load trained model
        model = YOLO(model_path, task='relation')
        
        # Validation arguments
        val_args = {
            'data': config_path,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'project': 'runs/relation',
            'name': 'test_relation_validation',
            'exist_ok': True,
        }
        
        # Run validation
        results = model.val(**val_args)
        
        print("✅ Relation validation completed successfully!")
        print(f"Validation results: {results}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Error during relation validation: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test YOLO Relation Detection Training')
    parser.add_argument('--config', type=str, default='ultralytics/cfg/datasets/PSG.yaml',
                       help='Path to PSG dataset configuration file')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only test integration')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model for validation test')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 YOLO Relation Detection Training Test")
    print("=" * 60)
    
    # Step 1: Check integration
    integration_ok = check_relation_task_integration()
    if not integration_ok:
        print("\n❌ Integration check failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Check component imports
    components_ok = check_relation_components()
    if not components_ok:
        print("\n❌ Component import check failed. Exiting.")
        sys.exit(1)
    
    # Step 3: Check PSG config
    config_result = check_psg_config()
    if isinstance(config_result, tuple):
        config_ok, config = config_result
    else:
        config_ok = config_result
        config = None
    
    if not config_ok:
        print("\n❌ PSG config check failed. Exiting.")
        sys.exit(1)
    
    # Step 4: Test training (if not skipped)
    if not args.skip_training:
        training_ok, training_results = test_relation_training(
            config_path=args.config,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz
        )
        
        if not training_ok:
            print("\n❌ Training test failed.")
            sys.exit(1)
    
    # Step 5: Test validation (if model path provided)
    if args.model_path:
        validation_ok, validation_results = test_relation_validation(
            model_path=args.model_path,
            config_path=args.config
        )
        
        if not validation_ok:
            print("\n❌ Validation test failed.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("✅ YOLO Relation Detection integration is working correctly")
    print("=" * 60)


if __name__ == "__main__":
    main()
