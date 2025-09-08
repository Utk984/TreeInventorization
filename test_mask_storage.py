#!/usr/bin/env python3
"""
Test script to demonstrate the new mask storage system.
This script shows how to load and work with masks stored in JSON format.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.mask_serialization import load_panorama_masks, deserialize_ultralytics_mask
from config import Config

def test_mask_loading():
    """Test loading masks from JSON files."""
    config = Config()
    
    # Check if mask directory exists and has files
    mask_dir = config.MASK_DIR
    if not os.path.exists(mask_dir):
        print(f"❌ Mask directory does not exist: {mask_dir}")
        return
    
    # Find JSON files in the mask directory
    json_files = [f for f in os.listdir(mask_dir) if f.endswith('_masks.json')]
    
    if not json_files:
        print(f"ℹ️ No mask JSON files found in {mask_dir}")
        print("Run the pipeline first to generate mask files.")
        return
    
    print(f"📁 Found {len(json_files)} mask JSON files:")
    for json_file in json_files[:3]:  # Show first 3 files
        print(f"  - {json_file}")
    
    # Load and examine the first file
    first_file = os.path.join(mask_dir, json_files[0])
    print(f"\n🔍 Examining {json_files[0]}:")
    
    try:
        mask_data = load_panorama_masks(first_file)
        
        print(f"  Panorama ID: {mask_data.get('pano_id', 'Unknown')}")
        print(f"  Total views: {mask_data.get('metadata', {}).get('total_views', 0)}")
        print(f"  Total masks: {mask_data.get('metadata', {}).get('total_masks', 0)}")
        
        # Show view structure
        views = mask_data.get('views', {})
        print(f"\n📋 Views in this panorama:")
        for view_name, view_masks in views.items():
            print(f"  {view_name}: {len(view_masks)} masks")
            
            # Show details of first mask in this view
            if view_masks:
                first_mask = view_masks[0]
                print(f"    First mask details:")
                print(f"      Tree index: {first_mask.get('tree_index', 'Unknown')}")
                print(f"      Image path: {first_mask.get('image_path', 'Unknown')}")
                print(f"      Confidence: {first_mask.get('confidence', 'Unknown')}")
                
                mask_data_obj = first_mask.get('mask_data', {})
                print(f"      Original shape: {mask_data_obj.get('orig_shape', 'Unknown')}")
                print(f"      Has xy data: {bool(mask_data_obj.get('xy'))}")
                print(f"      Has xyn data: {bool(mask_data_obj.get('xyn'))}")
                print(f"      Has dense data: {bool(mask_data_obj.get('data'))}")
        
        print(f"\n✅ Successfully loaded mask data from {json_files[0]}")
        
    except Exception as e:
        print(f"❌ Error loading mask data: {str(e)}")

def test_mask_deserialization():
    """Test deserializing a mask back to usable format."""
    config = Config()
    mask_dir = config.MASK_DIR
    
    # Find a JSON file
    json_files = [f for f in os.listdir(mask_dir) if f.endswith('_masks.json')]
    if not json_files:
        print("❌ No mask JSON files found for deserialization test")
        return
    
    first_file = os.path.join(mask_dir, json_files[0])
    
    try:
        mask_data = load_panorama_masks(first_file)
        
        # Get first mask from first view
        views = mask_data.get('views', {})
        if not views:
            print("❌ No views found in mask data")
            return
        
        first_view = list(views.values())[0]
        if not first_view:
            print("❌ No masks found in first view")
            return
        
        first_mask_info = first_view[0]
        mask_data_obj = first_mask_info['mask_data']
        
        print(f"🔄 Deserializing mask from {first_mask_info['tree_index']}:")
        
        # Deserialize the mask
        deserialized_mask = deserialize_ultralytics_mask(mask_data_obj)
        
        print(f"  Original shape: {deserialized_mask.get('orig_shape', 'Unknown')}")
        print(f"  XY data type: {type(deserialized_mask.get('xy', None))}")
        print(f"  XY data length: {len(deserialized_mask.get('xy', []))}")
        
        if deserialized_mask.get('xy'):
            xy_data = deserialized_mask['xy'][0]
            print(f"  First polygon points: {len(xy_data)} points")
            print(f"  First few points: {xy_data[:3].tolist() if hasattr(xy_data, 'tolist') else xy_data[:3]}")
        
        print(f"✅ Successfully deserialized mask")
        
    except Exception as e:
        print(f"❌ Error deserializing mask: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing Mask Storage System")
    print("=" * 50)
    
    test_mask_loading()
    print("\n" + "=" * 50)
    test_mask_deserialization()
    
    print("\n🎉 Test completed!")
