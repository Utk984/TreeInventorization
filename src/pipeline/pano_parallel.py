import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from streetlevel import streetview
import asyncio
import logging
import time
import re
from config import Config
from aiohttp import ClientSession
from src.inference.segment import detect_trunks, detect_trees
from src.utils.unwrap import divide_panorama
from src.utils.masks import add_masks, remove_duplicates, make_image, serialize_ultralytics_mask, save_panorama_masks
from src.utils.transformation import get_point
from src.utils.geodesic import localize_pixel_with_depth, get_depth_at_pixel
from concurrent.futures import ThreadPoolExecutor
import aiofiles

# Configure logger for pipeline
logger = logging.getLogger(__name__)

IO_EXECUTOR = ThreadPoolExecutor(max_workers=4)  # Increased for parallel processing

async def fetch_pano_by_id(pano_id: str, session: ClientSession):
    """Fetch panorama by ID with comprehensive logging."""
    logger.debug(f"🔍 Fetching panorama: {pano_id}")
    fetch_start_time = time.time()
    
    try:
        pano = await streetview.find_panorama_by_id_async(
            pano_id, session, download_depth=True
        )
        
        if pano is None:
            logger.warning(f"⚠️ No panorama found for {pano_id}")
            return None, None

        if pano.depth is None:
            logger.warning(f"⚠️ No depth map for {pano_id}")
            return None, None
            
        logger.debug(f"📍 Panorama found: {pano_id} at ({pano.lat}, {pano.lon})")
        
        # Fetch RGB image
        rgb = await streetview.get_panorama_async(pano, session) 
        rgb_array = np.array(rgb)
        
        fetch_time = time.time() - fetch_start_time
        logger.debug(f"✅ Panorama {pano_id} fetched in {fetch_time:.2f}s - Shape: {rgb_array.shape}")
        
        return pano, rgb_array
        
    except Exception as e:
        logger.error(f"❌ Error fetching panorama {pano_id}: {str(e)}")
        return None, None

def process_view(config: Config, view, tree_data, pano, image, theta, i):
    """Process a single view with detailed logging."""
    logger.debug(f"🔄 Processing view {i} at theta={theta}°")
    trees = []
    tree_masks = []  # List of mask objects corresponding to trees
    tree_count = 0
    
    try:
        for j, tree in enumerate(tree_data):
            masks = tree.masks
            boxes = tree.boxes
            
            if masks is not None:
                logger.debug(f"🌳 Processing {len(masks)} trees in view {i}")
                
                for k, mask in enumerate(masks):
                    tree_count += 1
                    image_path = os.path.join(config.VIEW_DIR, f"{pano.id}_view{i}_tree{j}_box{k}.jpg")
                    conf = boxes[k].conf.item()
                    
                    logger.debug(f"Processing tree {j}-{k} with confidence {conf:.3f}")

                    try:
                        orig_point, pers_point = get_point(mask, theta, pano, config.HEIGHT, config.WIDTH, config.FOV)
                        # Get distance from depth map
                        W, H = image.shape[1], image.shape[0]
                        
                        distance_pano = get_depth_at_pixel(pano.depth, orig_point[0], orig_point[1], W, H, flipped=True)
                        if distance_pano is None:
                            logger.warning(f"⚠️ No depth map for {pano.id} at {orig_point[0]}, {orig_point[1]}")
                            continue
                        if distance_pano > 15:
                            logger.warning(f"⚠️ Distance too far for {pano.id} at {orig_point[0]}, {orig_point[1]}")
                            continue
                        lat_pano, lon_pano = localize_pixel_with_depth(pano, orig_point[0], orig_point[1], W, H, distance_pano)

                        # Use pano depth directly (no depth model)
                        distance_calibrated = distance_pano
                        lat_model, lon_model = lat_pano, lon_pano
                        
                        logger.info(f"Pano distance: {distance_pano:.2f}m")
                        
                        # Submit image creation to thread pool
                        IO_EXECUTOR.submit(
                            make_image, view, boxes[k], mask, image_path
                        )
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing tree {k} in view {i}: {e}")
                        continue

                    tree = {
                        "image_path": image_path,
                        "pano_id": pano.id,
                        "stview_lat": pano.lat,
                        "stview_lng": pano.lon,
                        "tree_lat_model": lat_model,
                        "tree_lng_model": lon_model,
                        "tree_lat": lat_pano,
                        "tree_lng": lon_pano,
                        "image_x": float(orig_point[0]),
                        "image_y": float(orig_point[1]),
                        "theta": theta,
                        "conf": conf,
                        "distance_model": distance_calibrated,
                        "distance_pano": distance_pano,
                    }
                    trees.append(tree)
                    tree_masks.append(mask)  # Keep mask separately
        
        return trees, tree_masks
        
    except Exception as e:
        logger.error(f"❌ Error processing view {i}: {str(e)}")
        return [], []

async def process_single_panorama(pano_id: str, config: Config, tree_model, session: ClientSession, semaphore: asyncio.Semaphore):
    """Process a single panorama with semaphore for concurrency control."""
    async with semaphore:
        pano_start_time = time.time()
        
        try:
            # Fetch panorama
            pano, image = await fetch_pano_by_id(pano_id, session)
            
            if pano is None or image is None:
                logger.warning(f"⏭️ Skipping panorama {pano_id} due to fetch failure")
                return None
            
            logger.info(f"🔄 Processing panorama: {pano.id}")
            
            # Generate perspective views
            logger.info("🔄 Generating perspective views")
            view_start_time = time.time()
            views = divide_panorama(image, config.HEIGHT, config.WIDTH, config.FOV)
            view_time = time.time() - view_start_time
            logger.info(f"✅ Perspective Generation completed in {view_time:.3f}s")

            # Process each view
            trees = []
            all_masks = []  # List of mask objects for deduplication
            view_processing_start = time.time()
            for i, (view, theta) in enumerate(views):
                logger.debug(f"🔍 Detecting trunks in view {i}")
                tree_data = detect_trunks(view, tree_model, config.DEVICE)
                view_trees, tree_masks = process_view(config, view, tree_data, pano, image, theta, i)
                trees.extend(view_trees)
                all_masks.extend(tree_masks)  # Collect masks for deduplication
            
            view_processing_time = time.time() - view_processing_start
            logger.info(f"✅ View Processing completed in {view_processing_time:.3f}s")

            if not trees:
                logger.info(f"🌳 No trunks found in panorama {pano.id}")
                return None

            logger.info(f"🌳 Found {len(trees)} trunks in panorama {pano.id}")
            
            # Remove duplicates
            logger.info("🔄 Removing duplicate detections")
            dedup_start_time = time.time()
            
            df_part, masks_part = remove_duplicates(
                pd.DataFrame(trees),
                all_masks,  # Pass masks separately
                image.shape[1], image.shape[0],
                config.HEIGHT, config.WIDTH, config.FOV
            )
            dedup_time = time.time() - dedup_start_time
            logger.info(f"✅ Deduplication completed in {dedup_time:.3f}s - {len(df_part)} trees remaining")

            # Serialize only the kept masks for JSON storage
            if masks_part:
                logger.info(f"💾 Serializing {len(masks_part)} kept masks for JSON storage")
                serialization_start = time.time()
                
                # Group masks by view using the actual view number from image_path
                view_masks = {}
                for idx, mask in enumerate(masks_part):
                    if mask is not None and idx < len(df_part):
                        row = df_part.iloc[idx]
                        image_path = row.get('image_path', '')
                        
                        # Extract view number from image_path (e.g., "view3" -> 3)
                        view_num = None
                        if 'view' in image_path:
                            try:
                                # Find "view" followed by digits
                                match = re.search(r'view(\d+)', image_path)
                                if match:
                                    view_num = int(match.group(1))
                            except:
                                pass
                        
                        if view_num is not None:
                            view_name = f"view_{view_num}"
                            
                            if view_name not in view_masks:
                                view_masks[view_name] = []
                            
                            # Serialize the mask
                            serialized_mask = serialize_ultralytics_mask(mask)
                            view_masks[view_name].append({
                                "tree_index": f"{idx}-0",  # Simplified index
                                "image_path": image_path,
                                "confidence": float(row.get('conf', 0.0)),
                                "mask_data": serialized_mask
                            })
                        else:
                            logger.warning(f"⚠️ Could not extract view number from path: {image_path}")
                
                if view_masks:
                    mask_json_path = save_panorama_masks(pano.id, view_masks, config)
                    serialization_time = time.time() - serialization_start
                    logger.info(f"✅ Masks Serialization completed in {serialization_time:.3f}s to {mask_json_path}")
                else:
                    mask_json_path = None
            else:
                mask_json_path = None

            # Save full panorama with masks
            def _save_full(img=image.copy(), td=df_part.copy(), pid=pano.id, mask_path=mask_json_path):
                logger.debug(f"💾 Saving full panorama with masks: {pid}")
                full = add_masks(img, td, config.HEIGHT, config.WIDTH, config.FOV, mask_path)
                full_path = os.path.join(config.FULL_DIR, f"{pid}.jpg")
                cv2.imwrite(full_path, full)
                logger.debug(f"✅ Full Panorama saved: {full_path}")
                
                if not config.SAVE_MASK_JSON:
                    os.remove(mask_json_path)

            IO_EXECUTOR.submit(_save_full)

            # Save results to CSV (thread-safe)
            logger.info("💾 Saving results to CSV")
            save_start = time.time()
            
            # Use aiofiles for async file writing
            async with aiofiles.open(config.OUTPUT_CSV, 'a') as f:
                output_exists = os.path.exists(config.OUTPUT_CSV)
                if not output_exists or os.path.getsize(config.OUTPUT_CSV) == 0:
                    # Write header
                    await f.write(df_part.to_csv(index=False, lineterminator="\n"))
                else:
                    # Write data without header
                    await f.write(df_part.to_csv(index=False, header=False, lineterminator="\n"))
            
            save_time = time.time() - save_start
            logger.info(f"✅ Results CSV saved in {save_time:.3f}s")

            pano_time = time.time() - pano_start_time
            logger.info(f"✅ Panorama {pano.id} Processing completed in {pano_time:.3f}s")
            
            return {
                'pano_id': pano.id,
                'trees_found': len(df_part),
                'processing_time': pano_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing panorama {pano_id}: {str(e)}")
            logger.exception("Full traceback:")
            return None

async def process_panoramas_parallel(config: Config, tree_model, max_concurrent=3):
    """Main panorama processing function with parallel processing."""
    logger.info("🚀 Starting parallel panorama processing pipeline")
    pipeline_start_time = time.time()
    
    try:
        # Load panorama IDs
        logger.info(f"📋 Loading panorama IDs from: {config.PANORAMA_CSV}")
        pano_ids = pd.read_csv(config.PANORAMA_CSV)["pano_id"].tolist()
        
        # Create semaphore to limit concurrent panoramas
        semaphore = asyncio.Semaphore(max_concurrent)
        
        processed_count = 0
        skipped_count = 0
        error_count = 0

        async with ClientSession() as session:
            logger.info("🌐 HTTP session started")
            
            # Create tasks for all panoramas
            tasks = [
                process_single_panorama(pano_id, config, tree_model, session, semaphore)
                for pano_id in pano_ids
            ]
            
            # Process panoramas in parallel with progress bar
            logger.info(f"🔄 Processing {len(pano_ids)} panoramas with max {max_concurrent} concurrent")
            
            # Use asyncio.as_completed to get results as they finish
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing panoramas"):
                result = await coro
                
                if result is None:
                    skipped_count += 1
                else:
                    processed_count += 1
                    logger.info(f"✅ Completed {result['pano_id']}: {result['trees_found']} trees in {result['processing_time']:.2f}s")

        logger.info("🛑 Shutting down IO executor")
        IO_EXECUTOR.shutdown(wait=True)

        # Final statistics
        total_time = time.time() - pipeline_start_time
        
        logger.info("=" * 60)
        logger.info("📊 PARALLEL PIPELINE STATISTICS")
        logger.info(f"Total panoramas: {len(pano_ids)}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Skipped: {skipped_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Total pipeline time: {total_time:.3f}s")
        logger.info(f"Average time per panorama: {total_time/len(pano_ids):.3f}s")
        logger.info(f"Speedup factor: {len(pano_ids) * 5 / total_time:.2f}x")  # Assuming 5s per pano
        logger.info("=" * 60)
        logger.info("✅ Parallel pipeline finished successfully!")
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise
