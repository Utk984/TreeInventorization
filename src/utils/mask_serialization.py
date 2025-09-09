import json
import numpy as np
import torch
import logging
from typing import Dict, Any, List
import os
import cv2

# Configure logger for mask serialization
logger = logging.getLogger(__name__)

try:
    from pycocotools import mask as maskUtils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    logger.warning("pycocotools not installed. Install with: pip install pycocotools")

def serialize_ultralytics_mask(mask_obj) -> Dict[str, Any]:
    """
    Serialize an Ultralytics mask object to a JSON-serializable dictionary using RLE encoding.
    This reduces storage size by 100-1000x compared to storing polygon coordinates.
    
    Args:
        mask_obj: Ultralytics mask object
        
    Returns:
        Dict containing serialized mask data with RLE encoding
    """
    try:
        mask_data = {
            "orig_shape": mask_obj.orig_shape if hasattr(mask_obj, 'orig_shape') else None,
            "encoding": "rle" if HAS_PYCOCOTOOLS else "polygon",
        }
        
        if HAS_PYCOCOTOOLS and hasattr(mask_obj, 'xy') and mask_obj.xy is not None and len(mask_obj.xy) > 0:
            # Use RLE encoding for compact storage
            try:
                # Get the mask's bounding box and dimensions
                polygon = mask_obj.xy[0]
                if isinstance(polygon, torch.Tensor):
                    polygon = polygon.detach().cpu().numpy()
                
                # Get image dimensions from orig_shape or estimate from polygon
                if hasattr(mask_obj, 'orig_shape') and mask_obj.orig_shape:
                    h, w = mask_obj.orig_shape[:2]
                else:
                    # Estimate from polygon bounds
                    h = int(np.max(polygon[:, 1])) + 1
                    w = int(np.max(polygon[:, 0])) + 1
                
                # Create binary mask from polygon
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                pts = polygon.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(binary_mask, [pts], 1)
                
                # Encode to RLE (COCO format)
                rle = maskUtils.encode(np.asfortranarray(binary_mask))
                rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string for JSON
                
                mask_data["rle"] = rle
                mask_data["bbox"] = [
                    float(np.min(polygon[:, 0])),  # x_min
                    float(np.min(polygon[:, 1])),  # y_min
                    float(np.max(polygon[:, 0])),  # x_max
                    float(np.max(polygon[:, 1]))   # y_max
                ]
                
                logger.debug(f"✅ RLE encoded mask: shape={h}x{w}, RLE size={len(rle['counts'])} chars")
                
            except Exception as e:
                # Fallback to polygon storage if RLE fails
                logger.warning(f"⚠️ RLE encoding failed, falling back to polygon: {str(e)}")
                mask_data["encoding"] = "polygon"
                mask_data["xy"] = [polygon.tolist() if isinstance(polygon, np.ndarray) else polygon 
                                   for polygon in mask_obj.xy]
        else:
            # Fallback: store polygon coordinates (original method)
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
    Handles both RLE encoded masks and polygon format.
    
    Args:
        mask_data: Dictionary containing serialized mask data
        
    Returns:
        Dictionary with mask attributes including polygon coordinates
    """
    try:
        mask_obj = {}
        
        if mask_data.get("orig_shape"):
            mask_obj["orig_shape"] = tuple(mask_data["orig_shape"])
        
        # Handle RLE encoded masks
        if mask_data.get("encoding") == "rle" and mask_data.get("rle") and HAS_PYCOCOTOOLS:
            try:
                rle = mask_data["rle"]
                # Decode RLE to binary mask
                if isinstance(rle['counts'], str):
                    rle['counts'] = rle['counts'].encode('utf-8')
                binary_mask = maskUtils.decode(rle)
                
                # Find contours to get polygon coordinates
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour (main mask)
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Convert to xy format
                    polygon = largest_contour.squeeze().astype(np.float32)
                    if polygon.ndim == 2 and polygon.shape[0] > 2:
                        mask_obj["xy"] = [polygon]
                    else:
                        # Fallback if contour is too small
                        logger.warning("⚠️ Contour too small, using bbox")
                        if mask_data.get("bbox"):
                            x1, y1, x2, y2 = mask_data["bbox"]
                            mask_obj["xy"] = [np.array([
                                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                            ], dtype=np.float32)]
                
                # Store binary mask as well
                mask_obj["data"] = binary_mask.astype(np.float32)
                
                logger.debug(f"✅ Decoded RLE mask to polygon with {len(mask_obj.get('xy', [[]])[0])} points")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to decode RLE, trying polygon fallback: {str(e)}")
                # Try polygon fallback
                if mask_data.get("xy"):
                    mask_obj["xy"] = [np.array(polygon, dtype=np.float32) for polygon in mask_data["xy"]]
        
        # Handle polygon format (fallback or original)
        elif mask_data.get("xy"):
            mask_obj["xy"] = [np.array(polygon, dtype=np.float32) for polygon in mask_data["xy"]]
        
        if mask_data.get("xyn"):
            mask_obj["xyn"] = [np.array(polygon, dtype=np.float32) for polygon in mask_data["xyn"]]
        
        if mask_data.get("data") and "data" not in mask_obj:
            mask_obj["data"] = np.array(mask_data["data"], dtype=np.float32)
        
        return mask_obj
        
    except Exception as e:
        logger.error(f"❌ Error deserializing mask: {str(e)}")
        raise
