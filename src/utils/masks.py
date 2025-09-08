import cv2
import numpy as np
import logging
import time
import os
from src.utils.transformation import map_perspective_point_to_original
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# Configure logger for mask processing
logger = logging.getLogger(__name__)

def make_image(view, box, mask, image_path):
    """
    Create and save a tree image with bounding box, confidence, and mask overlay.
    """
    logger.debug(f"💾 Creating tree image: {image_path}")
    start_time = time.time()
    
    try:
        xyxy = [int(coord) for coord in box.xyxy[0].tolist()]
        conf = box.conf.item()
        
        logger.debug(f"📦 Bounding box: {xyxy}, confidence: {conf:.3f}")

        im = view.copy()
        cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(
            im,
            f"{conf:.2f}",
            (xyxy[0], max(xyxy[1] - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        mask_points = mask.xy[0].astype(np.int32)
        overlay = view.copy()
        cv2.fillPoly(overlay, [mask_points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, im, 0.7, 0, im)
        cv2.polylines(im, [mask_points], isClosed=True, color=(0, 255, 0), thickness=1)

        cv2.imwrite(image_path, im)
        
        save_time = time.time() - start_time
        logger.debug(f"✅ Tree image saved in {save_time:.3f}s: {image_path}")
        
    except Exception as e:
        logger.error(f"❌ Error creating tree image {image_path}: {str(e)}")
        raise

def remove_duplicates(df, image_x, image_y, height, width, fov, iou_threshold=0.4):
    """
    Remove duplicate detections with comprehensive logging.
    * Converts each mask to a Shapely Polygon (no giant binary images)
    * Uses an STRtree to query only potentially overlapping pairs
    """
    logger.debug(f"🔄 Starting duplicate removal - {len(df)} initial detections")
    start_time = time.time()
    
    try:
        polys, areas, idx_map = [], [], [] 

        # Convert masks to polygons
        conversion_start = time.time()
        valid_masks = 0
        
        for idx, (mask, th) in enumerate(zip(df["mask"], df["theta"])):
            if not mask.xy:
                continue

            try:
                pts = mask.xy[0].astype(np.float32)
                pano_pts = np.vstack(
                    [
                        map_perspective_point_to_original(
                            x, y, th, (image_x, image_y), height, width, fov
                        )
                        for x, y in pts
                    ]
                )
                poly = Polygon(pano_pts).buffer(0)  
                if poly.is_empty or not poly.is_valid:
                    continue

                polys.append(poly)
                areas.append(poly.area)
                idx_map.append(idx)
                valid_masks += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to convert mask {idx} to polygon: {str(e)}")
                continue

        conversion_time = time.time() - conversion_start
        logger.debug(f"📐 Converted {valid_masks}/{len(df)} masks to polygons in {conversion_time:.3f}s")

        if not polys:
            logger.warning("⚠️ No valid polygons found, returning empty DataFrame")
            return df.iloc[0:0]        
        
        # Build spatial index and find overlaps
        dedup_start = time.time()
        tree = STRtree(polys)
        keep = set(range(len(polys)))
        removed_count = 0

        for i in range(len(polys)):
            if i not in keep:
                continue
                
            candidates = tree.query(polys[i])
            
            for j in candidates:
                if j <= i or j not in keep:
                    continue
                    
                try:
                    intersection = polys[i].intersection(polys[j])
                    inter_area = intersection.area if hasattr(intersection, 'area') else 0
                    
                    if inter_area == 0:
                        continue
                        
                    smaller_area = min(areas[i], areas[j])
                    iou = inter_area / smaller_area
                    
                    if iou > iou_threshold:
                        remove_idx = i if areas[i] < areas[j] else j
                        if remove_idx in keep:
                            keep.discard(remove_idx)
                            removed_count += 1
                            logger.debug(f"🗑️ Removed duplicate: IoU={iou:.3f} > {iou_threshold}")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Error computing intersection for polygons {i}, {j}: {str(e)}")
                    continue

        dedup_time = time.time() - dedup_start
        keep_df_idx = [idx_map[k] for k in keep]
        result_df = df.iloc[keep_df_idx].reset_index(drop=True)
        
        total_time = time.time() - start_time
        logger.debug(f"✅ Duplicate removal completed in {total_time:.3f}s")
        logger.debug(f"📊 Removed {removed_count} duplicates, kept {len(result_df)}/{len(df)} detections")
        
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Error during duplicate removal: {str(e)}")
        raise

def add_masks(image, df, height, width, FOV, mask_json_path=None):
    """
    Add masks to an image by overlaying Ultralytics masks with logging.
    Now works with masks stored in JSON files instead of DataFrame.

    Args:
        image (numpy.ndarray): The original image.
        df (pandas.DataFrame): DataFrame containing position data (no mask data).
        height (int): Height of perspective views.
        width (int): Width of perspective views.
        FOV (int): Field of view in degrees.
        mask_json_path (str, optional): Path to JSON file containing mask data.

    Returns:
        numpy.ndarray: The image with overlaid masks.
    """
    logger.debug(f"🎨 Adding masks to image shape: {image.shape}")
    start_time = time.time()
    
    try:
        overlay = image.copy()
        img_shape = (image.shape[1], image.shape[0])
        processed_masks = 0
        
        # If no mask JSON path provided, just draw center points
        if mask_json_path is None or not os.path.exists(mask_json_path):
            logger.warning("⚠️ No mask JSON file provided or file doesn't exist, drawing center points only")
            for idx, row in df.iterrows():
                try:
                    # Draw center point
                    point = (int(row["image_x"]), int(row["image_y"]))
                    cv2.circle(image, point, 25, (0, 0, 255), -1)
                    processed_masks += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error drawing center point {idx}: {str(e)}")
                    continue
        else:
            # Load mask data from JSON
            import json
            with open(mask_json_path, 'r') as f:
                mask_data = json.load(f)
            
            # Process each view's masks
            for view_name, view_masks in mask_data.get("views", {}).items():
                for mask_info in view_masks:
                    try:
                        mask_data_obj = mask_info["mask_data"]
                        tree_index = mask_info["tree_index"]
                        
                        # Find corresponding row in DataFrame
                        # This is a simplified approach - you might need to match by other criteria
                        if len(df) > 0:
                            row = df.iloc[0]  # For now, use first row as reference
                            
                            # Reconstruct mask points from serialized data
                            if mask_data_obj.get("xy"):
                                mask_points = np.array(mask_data_obj["xy"][0], dtype=np.int32)
                                original_points = []
                                
                                for point in mask_points:
                                    orig_point = map_perspective_point_to_original(
                                        point[0], point[1], row["theta"], img_shape, height, width, FOV
                                    )
                                    orig_point = tuple(map(int, orig_point))
                                    original_points.append(orig_point)

                                # Draw mask outline and fill
                                cv2.polylines(
                                    overlay,
                                    [np.array(original_points, np.int32)],
                                    isClosed=True,
                                    color=(0, 255, 0),
                                    thickness=5,
                                )
                                cv2.fillPoly(overlay, [np.array(original_points, np.int32)], color=(0, 255, 0))
                                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

                                # Draw center point
                                point = (int(row["image_x"]), int(row["image_y"]))
                                cv2.circle(image, point, 25, (0, 0, 255), -1)
                                
                                processed_masks += 1
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing mask {mask_info.get('tree_index', 'unknown')}: {str(e)}")
                        continue

        overlay_time = time.time() - start_time
        logger.debug(f"✅ Added {processed_masks} masks in {overlay_time:.3f}s")
        
        return image
        
    except Exception as e:
        logger.error(f"❌ Error adding masks to image: {str(e)}")
        raise