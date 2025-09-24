import os
import pandas as pd
import asyncio
import logging
import time
from tqdm import tqdm

from config import Config
from src.pipeline.fetcher import fetch_panoramas_batch, create_optimized_session
from src.pipeline.processor import process_single_panorama, IO_EXECUTOR

logger = logging.getLogger(__name__)

async def inventorize(config: Config, tree_model, max_concurrent=3, chunk_size=15):
    """Simple fetch 15, process 15, repeat pipeline."""
    logger.info("🚀 Starting simple fetch-process pipeline")
    pipeline_start_time = time.time()
    
    try:
        # Load panorama IDs
        logger.info(f"📋 Loading panorama IDs from: {config.PANORAMA_CSV}")
        all_pano_ids = pd.read_csv(config.PANORAMA_CSV)["pano_id"].tolist()
        total_input_panoramas = len(all_pano_ids)
        
        # Check for already processed panoramas
        processed_pano_ids = set()
        if os.path.exists(config.OUTPUT_CSV) and os.path.getsize(config.OUTPUT_CSV) > 0:
            logger.info(f"📄 Found existing output CSV: {config.OUTPUT_CSV}")
            try:
                existing_df = pd.read_csv(config.OUTPUT_CSV)
                if 'pano_id' in existing_df.columns:
                    processed_pano_ids = set(existing_df['pano_id'].unique())
                    logger.info(f"✅ Found {len(processed_pano_ids)} already processed panoramas")
                else:
                    logger.warning("⚠️ Existing CSV doesn't have 'pano_id' column")
            except Exception as e:
                logger.warning(f"⚠️ Could not read existing CSV: {e}")
        else:
            logger.info("📝 No existing output CSV found, will create new one")
            # Initialize CSV file with headers
            empty_df = pd.DataFrame(columns=[
                'image_path', 'pano_id', 'stview_lat', 'stview_lng', 
                'tree_lat_model', 'tree_lng_model', 'tree_lat', 'tree_lng',
                'image_x', 'image_y', 'theta', 'conf', 'distance_model', 'distance_pano'
            ])
            empty_df.to_csv(config.OUTPUT_CSV, index=False)
            logger.info(f"✅ CSV headers initialized: {config.OUTPUT_CSV}")
        
        # Filter out already processed panoramas
        remaining_pano_ids = [pano_id for pano_id in all_pano_ids if pano_id not in processed_pano_ids]
        skipped_count = len(all_pano_ids) - len(remaining_pano_ids)
        
        logger.info(f"📊 Panorama Statistics:")
        logger.info(f"   Total input panoramas: {total_input_panoramas}")
        logger.info(f"   Already processed: {skipped_count}")
        logger.info(f"   Remaining to process: {len(remaining_pano_ids)}")
        
        if len(remaining_pano_ids) == 0:
            logger.info("🎉 All panoramas have already been processed!")
            return
        
        pano_ids = remaining_pano_ids
        total_panoramas = len(pano_ids)
        
        # Create semaphore to limit concurrent panoramas
        semaphore = asyncio.Semaphore(max_concurrent)
        
        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Create optimized HTTP session
        async with create_optimized_session() as session:
            logger.info("🌐 Optimized HTTP session started with connection pooling")
            
            # Process in simple chunks: fetch 15, process 15, repeat
            num_chunks = (total_panoramas + chunk_size - 1) // chunk_size
            logger.info(f"📦 Processing {total_panoramas} panoramas in {num_chunks} chunks of {chunk_size}")
            
            # Create overall progress bar for chunks
            chunk_progress = tqdm(total=num_chunks, desc="Chunks", unit="chunk")
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_panoramas)
                chunk_pano_ids = pano_ids[start_idx:end_idx]
                
                # Step 1: Fetch this chunk
                logger.info(f"🔍 Fetching chunk {chunk_idx + 1}/{num_chunks}: panoramas {start_idx + 1}-{end_idx}")
                fetch_start_time = time.time()
                
                panorama_data = await fetch_panoramas_batch(chunk_pano_ids, session, chunk_size)
                
                fetch_time = time.time() - fetch_start_time
                logger.info(f"✅ Chunk {chunk_idx + 1} fetched in {fetch_time:.2f}s")
                
                # Step 2: Process this chunk
                logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{num_chunks}")
                process_start_time = time.time()
                
                # Create processing tasks for successfully fetched panoramas
                processing_tasks = []
                for pano_id in chunk_pano_ids:
                    if panorama_data[pano_id] is not None:
                        pano, image = panorama_data[pano_id]
                        task = process_single_panorama(
                            pano_id, pano, image, config, tree_model, semaphore
                        )
                        processing_tasks.append(task)
                    else:
                        skipped_count += 1
                
                # Process all panoramas in this chunk
                logger.info(f"🔄 Processing {len(processing_tasks)} panoramas from chunk {chunk_idx + 1}")
                
                for coro in asyncio.as_completed(processing_tasks):
                    result = await coro
                    
                    if result is None:
                        skipped_count += 1
                    else:
                        processed_count += 1
                        logger.info(f"✅ Completed {result['pano_id']}: {result['trees_found']} trees in {result['processing_time']:.2f}s")
                
                process_time = time.time() - process_start_time
                logger.info(f"✅ Chunk {chunk_idx + 1} processed in {process_time:.2f}s - Total: {processed_count}, Skipped: {skipped_count}")
                
                # Update overall chunk progress
                chunk_progress.update(1)
            
            # Close progress bar
            chunk_progress.close()

        logger.info("🛑 Shutting down IO executor")
        IO_EXECUTOR.shutdown(wait=True)

        # Final statistics
        total_time = time.time() - pipeline_start_time
        
        logger.info("=" * 60)
        logger.info("📊 SIMPLE PIPELINE STATISTICS")
        logger.info(f"Total input panoramas: {total_input_panoramas}")
        logger.info(f"Already processed (skipped): {skipped_count}")
        logger.info(f"Remaining panoramas: {total_panoramas}")
        logger.info(f"Successfully processed this run: {processed_count}")
        logger.info(f"Errors this run: {error_count}")
        logger.info(f"Total pipeline time: {total_time:.3f}s")
        if total_panoramas > 0:
            logger.info(f"Average time per panorama: {total_time/total_panoramas:.3f}s")
            logger.info(f"Speedup factor: {total_panoramas * 5 / total_time:.2f}x")
        logger.info("=" * 60)
        logger.info("✅ Simple pipeline finished successfully!")
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise
