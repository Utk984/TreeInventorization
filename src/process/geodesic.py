import math


def get_coordinates(pano, point, width, distance):
    center = width/2
    bearing_radians = pano.heading/center * point[0]
    R = 6371000

    # Calculate angular distance in radians
    distance /= R
    
    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(pano.lat)
    lng_rad = math.radians(pano.lon)
    
    # Calculate new latitude
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance) +
        math.cos(lat_rad) * math.sin(distance) * math.cos(bearing_radians)
    )
    
    # Calculate new longitude
    new_lng_rad = lng_rad + math.atan2(
        math.sin(bearing_radians) * math.sin(distance) * math.cos(lat_rad),
        math.cos(distance) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )
    
    # Convert back to degrees
    lat = math.degrees(new_lat_rad)
    lng = math.degrees(new_lng_rad)
    return lat, lng