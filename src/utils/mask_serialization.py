import json
import numpy as np
import torch
import logging
from typing import Dict, Any, List
import os

# Configure logger for mask serialization
logger = logging.getLogger(__name__)

def serialize_ultralytics_mask(mask_obj) -> Dict[str, Any]:
    """
    Serialize an Ultralytics mask object to a JSON-serializable dictionary.
    
    Args:
        mask_obj: Ultralytics mask object
        
    Returns:
        Dict containing serialized mask data
    """
    try:
        mask_data = {
            "orig_shape": mask_obj.orig_shape if hasattr(mask_obj, 'orig_shape') else None,
            "xy": None,
            "xyn": None,
            "data": None
        }
        
        # Serialize polygon coordinates (xy)
        if hasattr(mask_obj, 'xy') and mask_obj.xy is not None:
            xy_list = []
            for polygon in mask_obj.xy:
                if isinstance(polygon, np.ndarray):
                    xy_list.append(polygon.tolist())
                elif isinstance(polygon, torch.Tensor):
                    xy_list.append(polygon.detach().cpu().numpy().tolist())
                else:
                    xy_list.append(polygon)
            mask_data["xy"] = xy_list
        
        # Serialize normalized polygon coordinates (xyn)
        if hasattr(mask_obj, 'xyn') and mask_obj.xyn is not None:
            xyn_list = []
            for polygon in mask_obj.xyn:
                if isinstance(polygon, np.ndarray):
                    xyn_list.append(polygon.tolist())
                elif isinstance(polygon, torch.Tensor):
                    xyn_list.append(polygon.detach().cpu().numpy().tolist())
                else:
                    xyn_list.append(polygon)
            mask_data["xyn"] = xyn_list
        
        # Serialize dense mask data
        if hasattr(mask_obj, 'data') and mask_obj.data is not None:
            data = mask_obj.data
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            if isinstance(data, np.ndarray):
                # Convert to list for JSON serialization
                mask_data["data"] = data.tolist()
            else:
                mask_data["data"] = data
        
        logger.debug(f"✅ Successfully serialized mask with orig_shape: {mask_data['orig_shape']}")
        return mask_data
        
    except Exception as e:
        logger.error(f"❌ Error serializing mask: {str(e)}")
        raise

def save_panorama_masks(pano_id: str, view_masks: Dict[str, List[Dict[str, Any]]], config) -> str:
    """
    Save masks for a panorama in JSON format.
    
    Args:
        pano_id: Panorama ID
        view_masks: Dictionary with view paths as keys and list of mask data as values
        config: Configuration object with MASK_DIR
        
    Returns:
        Path to the saved JSON file
    """
    try:
        # Create mask directory if it doesn't exist
        os.makedirs(config.MASK_DIR, exist_ok=True)
        
        # Create JSON file path
        json_path = os.path.join(config.MASK_DIR, f"{pano_id}_masks.json")
        
        # Prepare data structure
        mask_data = {
            "pano_id": pano_id,
            "views": view_masks,
            "metadata": {
                "total_views": len(view_masks),
                "total_masks": sum(len(masks) for masks in view_masks.values())
            }
        }
        
        # Save to JSON file
        with open(json_path, 'w') as f:
            json.dump(mask_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {mask_data['metadata']['total_masks']} masks for panorama {pano_id} to {json_path}")
        return json_path
        
    except Exception as e:
        logger.error(f"❌ Error saving panorama masks for {pano_id}: {str(e)}")
        raise

def load_panorama_masks(json_path: str) -> Dict[str, Any]:
    """
    Load masks for a panorama from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary containing mask data
    """
    try:
        with open(json_path, 'r') as f:
            mask_data = json.load(f)
        
        logger.debug(f"✅ Loaded masks from {json_path}")
        return mask_data
        
    except Exception as e:
        logger.error(f"❌ Error loading masks from {json_path}: {str(e)}")
        raise

def deserialize_ultralytics_mask(mask_data: Dict[str, Any]):
    """
    Deserialize a mask from JSON data back to a format compatible with Ultralytics.
    Note: This creates a dictionary that mimics the Ultralytics mask object structure.
    
    Args:
        mask_data: Dictionary containing serialized mask data
        
    Returns:
        Dictionary with mask attributes
    """
    try:
        mask_obj = {}
        
        if mask_data.get("orig_shape"):
            mask_obj["orig_shape"] = tuple(mask_data["orig_shape"])
        
        if mask_data.get("xy"):
            mask_obj["xy"] = [np.array(polygon, dtype=np.float32) for polygon in mask_data["xy"]]
        
        if mask_data.get("xyn"):
            mask_obj["xyn"] = [np.array(polygon, dtype=np.float32) for polygon in mask_data["xyn"]]
        
        if mask_data.get("data"):
            mask_obj["data"] = np.array(mask_data["data"], dtype=np.float32)
        
        logger.debug(f"✅ Successfully deserialized mask")
        return mask_obj
        
    except Exception as e:
        logger.error(f"❌ Error deserializing mask: {str(e)}")
        raise
