import asyncio
import logging
import time
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from streetlevel import streetview
import numpy as np

logger = logging.getLogger(__name__)

async def fetch_pano_by_id(pano_id: str, session: ClientSession, max_retries: int = 3):
    """Fetch panorama by ID with retry logic and optimized networking."""
    logger.info(f"🔍 Fetching panorama: {pano_id}")
    fetch_start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            # Fetch panorama metadata and depth
            pano = await streetview.find_panorama_by_id_async(
                pano_id, session, download_depth=True
            )
            
            if pano is None:
                logger.warning(f"⚠️ No panorama found for {pano_id}")
                return None, None

            if pano.depth is None:
                logger.warning(f"⚠️ No depth map for {pano_id}")
                return None, None
                
            logger.info(f"📍 Panorama found: {pano_id} at ({pano.lat}, {pano.lon})")
            
            # Fetch RGB image with retry
            rgb = await streetview.get_panorama_async(pano, session) 
            rgb_array = np.array(rgb)
            
            fetch_time = time.time() - fetch_start_time
            logger.info(f"✅ Panorama {pano_id} fetched in {fetch_time:.2f}s - Shape: {rgb_array.shape}")
            
            return pano, rgb_array
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout fetching {pano_id} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
        except Exception as e:
            logger.warning(f"❌ Error fetching panorama {pano_id} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
    
    logger.error(f"❌ Failed to fetch panorama {pano_id} after {max_retries} attempts")
    return None, None

async def fetch_panoramas_batch(pano_ids: list, session: ClientSession, batch_size: int = 5):
    """Fetch multiple panoramas concurrently in batches."""
    logger.info(f"🔄 Fetching {len(pano_ids)} panoramas in batches of {batch_size}")
    
    all_panoramas = {}
    batch_start = 0
    
    while batch_start < len(pano_ids):
        batch_end = min(batch_start + batch_size, len(pano_ids))
        batch_ids = pano_ids[batch_start:batch_end]
        
        logger.info(f"📦 Fetching batch {batch_start//batch_size + 1}: {len(batch_ids)} panoramas")
        batch_start_time = time.time()
        
        # Create fetch tasks for this batch
        fetch_tasks = [
            fetch_pano_by_id(pano_id, session) 
            for pano_id in batch_ids
        ]
        
        # Execute batch fetch concurrently
        batch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(batch_results):
            pano_id = batch_ids[i]
            if isinstance(result, Exception):
                logger.error(f"❌ Failed to fetch {pano_id}: {result}")
                all_panoramas[pano_id] = None
            elif result[0] is None or result[1] is None:
                logger.warning(f"⚠️ No data for {pano_id}")
                all_panoramas[pano_id] = None
            else:
                pano, image = result
                all_panoramas[pano_id] = (pano, image)
        
        batch_time = time.time() - batch_start_time
        successful = sum(1 for r in batch_results if not isinstance(r, Exception) and r[0] is not None)
        logger.info(f"✅ Batch completed in {batch_time:.2f}s - {successful}/{len(batch_ids)} successful")
        
        batch_start = batch_end
    
    return all_panoramas

def create_optimized_session():
    """Create an optimized HTTP session with connection pooling."""
    connector = TCPConnector(
        limit=200,           # Total connection pool size
        limit_per_host=100,  # Connections per host
        ttl_dns_cache=300,   # DNS cache for 5 minutes
        use_dns_cache=True,
        keepalive_timeout=30, # Keep connections alive
        enable_cleanup_closed=True
    )
    
    timeout = ClientTimeout(
        total=60,           # Total timeout
        connect=10,         # Connection timeout
        sock_read=20        # Socket read timeout
    )
    
    return ClientSession(connector=connector, timeout=timeout)
