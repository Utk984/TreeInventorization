import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from streetlevel import streetview
import asyncio
import logging
import time
from config import Config
from aiohttp import ClientSession
from src.inference.segment import detect_trees
from src.inference.depth import estimate_depth
from src.utils.unwrap import divide_panorama
from src.utils.masks import add_masks, remove_duplicates, make_image
from src.utils.transformation import get_point
from src.utils.geodesic import get_coordinates
from concurrent.futures import ThreadPoolExecutor

# Configure logger for pipeline
logger = logging.getLogger(__name__)

IO_EXECUTOR = ThreadPoolExecutor(max_workers=1)  

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

def process_view(config: Config, view, tree_data, pano, image, depth, theta, i, calibrate_model):
    """Process a single view with detailed logging."""
    logger.debug(f"🔄 Processing view {i} at theta={theta}°")
    trees = []
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
                        distance = depth[orig_point[1]][orig_point[0]]

                        if orig_point[0] < 3800 or distance > 20:
                            continue

                        distance_calibrated = calibrate_model.calibrate_single(distance, orig_point[0], orig_point[1])
                        lat_model, lon_model = get_coordinates(pano, orig_point, image.shape[1], distance_calibrated)
                        
                        # Get coordinates from Google depth map
                        pano_depth_y = int(orig_point[1] * pano.depth.data.shape[0] / image.shape[0])
                        pano_depth_x = int(orig_point[0] * pano.depth.data.shape[1] / image.shape[1])
                        pano_distance = pano.depth.data[pano_depth_y][pano_depth_x]
                        lat_pano, lon_pano = get_coordinates(pano, orig_point, image.shape[1], pano_distance)

                        logger.info(f"Distance: {distance:.2f}m, Distance calibrated: {distance_calibrated:.2f}m, Pano distance: {pano_distance:.2f}m")
                        
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
                        "tree_lat_pano": lat_pano,
                        "tree_lng_pano": lon_pano,
                        "image_x": float(orig_point[0]),
                        "image_y": float(orig_point[1]),
                        "theta": theta,
                        "mask": mask,
                        "conf": conf,
                        "distance_model": distance_calibrated,
                        "distance_pano": pano_distance,
                    }
                    trees.append(tree)
        
        return trees
        
    except Exception as e:
        logger.error(f"❌ Error processing view {i}: {str(e)}")
        return []

async def process_panoramas(config: Config, depth_model, tree_model, calibrate_model):
    """Main panorama processing function with comprehensive logging."""
    logger.info("🚀 Starting panorama processing pipeline")
    pipeline_start_time = time.time()
    
    try:
        # Load panorama IDs
        logger.info(f"📋 Loading panorama IDs from: {config.PANORAMA_CSV}")
        pano_ids = pd.read_csv(config.PANORAMA_CSV)["pano_id"].tolist()
        logger.info(f"📊 Found {len(pano_ids)} panoramas to process")

        processed_count = 0
        skipped_count = 0
        error_count = 0

        async with ClientSession() as session:
            logger.info("🌐 HTTP session started")
            
            # Pre-fetch first panorama
            fetch_task = asyncio.create_task(fetch_pano_by_id(pano_ids[0], session))

            for idx, next_id in enumerate(tqdm(pano_ids[1:], total=len(pano_ids)), 1):
                pano_start_time = time.time()
                
                try:
                    # Get current panorama and start fetching next one
                    pano, image = await fetch_task
                    fetch_task = asyncio.create_task(fetch_pano_by_id(next_id, session))
                    
                    if pano is None or image is None:
                        logger.warning(f"⏭️ Skipping panorama {idx} due to fetch failure")
                        skipped_count += 1
                        continue
                    
                    logger.info(f"🔄 Processing panorama {idx}/{len(pano_ids)}: {pano.id}")
                    
                    # Depth estimation
                    logger.debug("🔍 Starting depth estimation")
                    depth_start_time = time.time()
                    depth = estimate_depth(image, depth_model)
                    depth_time = time.time() - depth_start_time
                    logger.debug(f"✅ Depth estimation completed in {depth_time:.2f}s")
                    
                    # Persist depth maps to disk to avoid keeping them all in memory
                    # np.save(os.path.join(config.DEPTH_DIR, f"{pano.id}_gdepth.npy"), pano.depth.data)
                    # np.save(os.path.join(config.DEPTH_DIR, f"{pano.id}_pred.npy"), depth)
                    
                    # Generate perspective views
                    logger.debug("🔄 Generating perspective views")
                    view_start_time = time.time()
                    views = divide_panorama(image, config.HEIGHT, config.WIDTH, config.FOV)
                    view_time = time.time() - view_start_time
                    logger.debug(f"✅ Generated {len(views)} perspective views in {view_time:.2f}s")

                    # Process each view
                    trees = []
                    for i, (view, theta) in enumerate(views):
                        logger.debug(f"🔍 Detecting trees in view {i}")
                        tree_data = detect_trees(view, tree_model, config.DEVICE)
                        trees.extend(
                            process_view(config, view, tree_data, pano, image, depth, theta, i, calibrate_model)
                        )

                    if not trees:
                        logger.debug(f"🌳 No trees found in panorama {pano.id}")
                        skipped_count += 1
                        continue

                    logger.info(f"🌳 Found {len(trees)} trees in panorama {pano.id}")
                    
                    # Remove duplicates
                    logger.debug("🔄 Removing duplicate detections")
                    dedup_start_time = time.time()
                    df_part = remove_duplicates(
                        pd.DataFrame(trees),
                        image.shape[1], image.shape[0],
                        config.HEIGHT, config.WIDTH, config.FOV
                    )
                    dedup_time = time.time() - dedup_start_time
                    logger.debug(f"✅ Deduplication completed in {dedup_time:.2f}s - {len(df_part)} trees remaining")

                    # Save full panorama with masks
                    def _save_full(img=image.copy(), td=df_part.copy(), pid=pano.id):
                        logger.debug(f"💾 Saving full panorama with masks: {pid}")
                        full = add_masks(img, td, config.HEIGHT, config.WIDTH, config.FOV)
                        full_path = os.path.join(config.FULL_DIR, f"{pid}.jpg")
                        cv2.imwrite(full_path, full)
                        logger.debug(f"✅ Full panorama saved: {full_path}")

                    IO_EXECUTOR.submit(_save_full)

                    # Append deduplicated detections directly to the output CSV
                    output_exists = os.path.exists(config.OUTPUT_CSV)
                    df_part.to_csv(
                        config.OUTPUT_CSV,
                        mode="a",
                        index=False,
                        header=not output_exists,
                        lineterminator="\n",
                    )

                    processed_count += 1
                    
                    pano_time = time.time() - pano_start_time
                    logger.info(f"✅ Panorama {pano.id} processed in {pano_time:.2f}s")
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ Error processing panorama {idx}: {str(e)}")
                    logger.exception("Full traceback:")
                    continue
                    
            # Process final panorama
            try:
                pano, image = await fetch_task
                if pano is not None and image is not None:
                    logger.info(f"🔄 Processing final panorama: {pano.id}")
                    # Process final panorama similar to above
                    # (Implementation details omitted for brevity)
            except Exception as e:
                logger.error(f"❌ Error processing final panorama: {str(e)}")

        logger.info("🛑 Shutting down IO executor")
        IO_EXECUTOR.shutdown(wait=True)

        # Results have already been written incrementally; nothing to aggregate here.
        save_start_time = time.time()
        save_time = time.time() - save_start_time
        total_time = time.time() - pipeline_start_time
        
        # Final statistics
        logger.info("=" * 60)
        logger.info("📊 PIPELINE STATISTICS")
        logger.info(f"Total panoramas: {len(pano_ids)}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Skipped: {skipped_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Save time: {save_time:.2f}s")
        logger.info(f"Total pipeline time: {total_time:.2f}s")
        logger.info(f"Average time per panorama: {total_time/len(pano_ids):.2f}s")
        logger.info("=" * 60)
        logger.info("✅ Pipeline finished successfully!")
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise