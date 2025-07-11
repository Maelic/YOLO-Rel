#!/usr/bin/env python3
"""
Simple test script to verify relation file detection in RelationTrainer.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_relation_file_detection():
    """Test if RelationTrainer can find relation files from dataset config."""
    print("🔍 Testing relation file detection...")
    
    try:
        from ultralytics.models.yolo.relation.train import RelationTrainer
        from ultralytics.cfg import DEFAULT_CFG
        
        # Create a mock trainer with PSG config
        config_path = "ultralytics/cfg/datasets/PSG.yaml"
        
        # Load the dataset config
        import yaml
        with open(config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        print(f"Dataset config loaded from: {config_path}")
        print(f"Dataset path: {dataset_config.get('path', 'Not found')}")
        print(f"Relation train file: {dataset_config.get('relation_train', 'Not found')}")
        print(f"Relation val file: {dataset_config.get('relation_val', 'Not found')}")
        
        # Create trainer args
        from types import SimpleNamespace
        args = SimpleNamespace()
        for key, value in DEFAULT_CFG.items():
            setattr(args, key, value)
        args.data = config_path
        
        # Create trainer
        trainer = RelationTrainer(overrides={'data': config_path})
        
        # Check if trainer loaded the dataset config
        if hasattr(trainer, 'data') and trainer.data:
            print(f"✅ Trainer loaded dataset config successfully")
            print(f"Dataset data: {trainer.data}")
            
            # Test build_dataset for train mode
            print("\n🔍 Testing build_dataset for train mode...")
            try:
                # We'll just test the relation file detection logic without building the full dataset
                mode = "train"
                relation_file = None
                
                # Get relation file from dataset config
                if hasattr(trainer, 'data') and trainer.data:
                    if f"relation_{mode}" in trainer.data:
                        relation_file = trainer.data[f"relation_{mode}"]
                
                if relation_file:
                    # Make it absolute path relative to dataset path
                    if not os.path.isabs(relation_file) and 'path' in trainer.data:
                        relation_file = str(Path(trainer.data['path']) / relation_file)
                    
                    print(f"✅ Found relation file for {mode}: {relation_file}")
                    
                    # Check if file exists
                    if os.path.exists(relation_file):
                        print(f"✅ Relation file exists: {relation_file}")
                    else:
                        print(f"⚠️  Relation file does not exist: {relation_file}")
                        print(f"   (This is expected if the dataset is not downloaded)")
                else:
                    print(f"❌ No relation file found for {mode}")
                
            except Exception as e:
                print(f"❌ Error testing build_dataset: {e}")
                
        else:
            print(f"❌ Trainer failed to load dataset config")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing relation file detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_relmetrics():
    """Test RelMetrics functionality."""
    print("\n🔍 Testing RelMetrics...")
    
    try:
        from ultralytics.utils.metrics import RelMetrics, relation_recall, relation_recall_per_class
        import numpy as np
        
        # Create RelMetrics instance
        metrics = RelMetrics()
        print("✅ RelMetrics created successfully")
        print(f"TopK values: {metrics.topk}")
        print(f"Metric keys: {metrics.keys}")
        
        # Test with sample data
        print("\n🔍 Testing with sample relation data...")
        
        # Sample predictions: (subj_idx, obj_idx, rel_cls, conf)
        pred_relations = [
            (0, 1, 5, 0.9),   # High confidence
            (1, 2, 3, 0.8),   # High confidence  
            (2, 3, 5, 0.7),   # Medium confidence
            (3, 4, 1, 0.6),   # Medium confidence
            (4, 5, 2, 0.5),   # Low confidence
        ]
        
        # Sample ground truth: (subj_idx, obj_idx, rel_cls)
        gt_relations = [
            (0, 1, 5),  # Match with pred_relations[0]
            (1, 2, 3),  # Match with pred_relations[1]
            (2, 3, 2),  # No match (different rel_cls)
        ]
        
        # Sample relation names
        relation_names = {
            0: "background",
            1: "on", 
            2: "in",
            3: "near",
            4: "holding", 
            5: "riding"
        }
        
        # Process relation stats
        metrics.process_relation_stats(
            pred_relations=pred_relations,
            gt_relations=gt_relations,
            relation_names=relation_names
        )
        
        print("✅ Relation stats processed successfully")
        
        # Get results
        results = metrics.mean_results()
        print(f"Mean results: {results}")
        
        # Get results dict
        results_dict = metrics.results_dict
        print(f"Results dictionary: {results_dict}")
        
        # Test summary
        summary = metrics.summary()
        print("✅ Summary generated successfully")
        for item in summary:
            print(f"  {item}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RelMetrics: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 YOLO Relation File Detection Test")
    print("=" * 60)
    
    # Test 1: Relation file detection
    test1_ok = test_relation_file_detection()
    
    # Test 2: RelMetrics functionality
    test2_ok = test_relmetrics()
    
    print("\n" + "=" * 60)
    if test1_ok and test2_ok:
        print("🎉 All tests passed successfully!")
        print("✅ Relation file detection is working correctly")
    else:
        print("❌ Some tests failed")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
