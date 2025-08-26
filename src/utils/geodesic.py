import math
import logging
import time

# Configure logger for geodesic calculations
logger = logging.getLogger(__name__)

def get_coordinates(pano, point, width, distance):
    """
    Calculate GPS coordinates from panorama position and detected tree location with logging.
    
    Args:
        pano: Panorama object with lat, lon, and heading
        point: (x, y) pixel coordinates of tree in equirectangular image
        width: Width of the equirectangular image
        distance: Distance to tree in meters
        
    Returns:
        tuple: (latitude, longitude) of the tree
    """
    logger.debug(f"🌍 Calculating GPS coordinates for point {point} at distance {distance:.2f}m")
    start_time = time.time()
    
    try:
        center = width / 2
        bearing_radians = pano.heading / center * point[0]
        R = 6371000  # Earth radius in meters

        # Calculate angular distance in radians
        angular_distance = distance / R
        
        # Convert latitude and longitude from degrees to radians
        lat_rad = math.radians(pano.lat)
        lng_rad = math.radians(pano.lon)
        
        logger.debug(f"📍 Panorama location: ({pano.lat:.6f}, {pano.lon:.6f})")
        logger.debug(f"🧭 Bearing: {math.degrees(bearing_radians):.2f}°, Angular distance: {angular_distance:.6f} rad")
        
        # Calculate new latitude using spherical trigonometry
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) +
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_radians)
        )
        
        # Calculate new longitude
        new_lng_rad = lng_rad + math.atan2(
            math.sin(bearing_radians) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        # Convert back to degrees
        lat = math.degrees(new_lat_rad)
        lng = math.degrees(new_lng_rad)
        
        calc_time = time.time() - start_time
        logger.debug(f"✅ GPS calculation completed in {calc_time:.3f}s")
        logger.debug(f"📍 Tree coordinates: ({lat:.6f}, {lng:.6f})")
        
        return lat, lng
        
    except Exception as e:
        logger.error(f"❌ Error calculating GPS coordinates: {str(e)}")
        logger.error(f"📊 Input parameters: point={point}, width={width}, distance={distance}")
        raise