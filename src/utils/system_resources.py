import os
import psutil
import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_system_resources() -> Tuple[int, int, int, int]:
    """
    Get system resource information.
    
    Returns:
        Tuple of (cpu_cores, ram_gb, gpu_memory_gb, available_ram_gb)
    """
    try:
        # CPU cores
        cpu_cores = os.cpu_count() or 1
        
        # RAM information
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total // (1024**3)
        available_ram_gb = memory.available // (1024**3)
        
        # GPU memory (if available)
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        logger.info(f"System resources detected:")
        logger.info(f"  CPU cores: {cpu_cores}")
        logger.info(f"  Total RAM: {total_ram_gb}GB")
        logger.info(f"  Available RAM: {available_ram_gb}GB")
        logger.info(f"  GPU memory: {gpu_memory_gb}GB")
        
        return cpu_cores, total_ram_gb, gpu_memory_gb, available_ram_gb
        
    except Exception as e:
        logger.error(f"Error detecting system resources: {e}")
        # Return conservative defaults
        return 4, 8, 0, 4

def calculate_optimal_concurrency(
    cpu_cores: int = None,
    available_ram_gb: int = None,
    gpu_memory_gb: int = None,
    min_concurrent: int = 1,
    max_concurrent: int = 32
) -> int:
    """
    Calculate optimal max_concurrent based on system resources.
    
    Args:
        cpu_cores: Number of CPU cores (auto-detected if None)
        available_ram_gb: Available RAM in GB (auto-detected if None)
        gpu_memory_gb: GPU memory in GB (auto-detected if None)
        min_concurrent: Minimum concurrent processes
        max_concurrent: Maximum concurrent processes (safety limit)
    
    Returns:
        Optimal number of concurrent processes
    """
    if cpu_cores is None or available_ram_gb is None or gpu_memory_gb is None:
        cpu_cores, _, gpu_memory_gb, available_ram_gb = get_system_resources()
    
    # Memory-based calculation
    # Each panorama processing needs approximately:
    # - 2-4GB RAM for image processing and model inference
    # - 1-2GB GPU memory for YOLO model (if GPU available)
    ram_based_limit = max(1, available_ram_gb // 3)  # Conservative: 3GB per process
    
    # CPU-based calculation
    # Leave some cores for system processes and I/O
    cpu_based_limit = max(1, cpu_cores // 2)  # Use half the cores
    
    # GPU-based calculation (if GPU available)
    gpu_based_limit = max_concurrent  # No GPU limit if no GPU
    if gpu_memory_gb > 0:
        # Each process needs ~1-2GB GPU memory
        gpu_based_limit = max(1, gpu_memory_gb // 2)
    
    # Take the most restrictive limit
    optimal_concurrent = min(ram_based_limit, cpu_based_limit, gpu_based_limit)
    
    # Apply safety bounds
    optimal_concurrent = max(min_concurrent, min(optimal_concurrent, max_concurrent))
    
    logger.info(f"Concurrency calculation:")
    logger.info(f"  RAM-based limit: {ram_based_limit} (based on {available_ram_gb}GB available)")
    logger.info(f"  CPU-based limit: {cpu_based_limit} (based on {cpu_cores} cores)")
    logger.info(f"  GPU-based limit: {gpu_based_limit} (based on {gpu_memory_gb}GB GPU memory)")
    logger.info(f"  Optimal concurrent: {optimal_concurrent}")
    
    return optimal_concurrent

def get_safe_concurrency_estimate() -> int:
    """
    Get a safe estimate of concurrent processes for panorama processing.
    This is a convenience function that auto-detects resources and calculates optimal concurrency.
    """
    return calculate_optimal_concurrency()
