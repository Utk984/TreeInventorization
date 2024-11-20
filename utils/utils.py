import math
from math import asin, atan2, cos, degrees, radians, sin

from streetlevel import streetview


def handle_model_output(
    image_x,
    image_y,
    theta,
    depth,
    heading,
    image_width,
    image_height,
    image_lat,
    image_lng,
):
    distance, direction, depth = calculate(
        image_x,
        image_y,
        abs(theta - 90),
        0,
        depth,
        image_width,
        image_height,
        heading,
    )

    if depth > 0 and distance > 0:
        lat, lng = move_in_heading(
            image_lat, image_lng, int(direction), distance / 1000
        )
        return lat, lng
    return None, None


def calculate_distance_and_direction(depth, pitch, yaw, heading):
    distance = depth * math.sin((180 - pitch) / 360)
    direction = yaw - 270 + heading
    if direction < 0:
        direction += 360
    return distance, direction


def calculate_image_pixel_coordinates(cal_yaw, cal_pitch, image_width, image_height):
    image_pixel_x = cal_yaw * (image_width / 360)
    image_pixel_y = cal_pitch * (image_height / 180)
    return image_pixel_x, image_pixel_y


def calculate_depth_indices(
    image_pixel_x, image_pixel_y, depth, image_width, image_height
):
    index_y = int(image_pixel_y * (depth.shape[0] / image_height))
    index_x = int(image_pixel_x * (depth.shape[1] / image_width)) * (-1)
    return depth[index_y][index_x]


def calculate(
    image_pixel_x,
    image_pixel_y,
    yaw,
    pitch,
    depth,
    image_width,
    image_height,
    heading,
):
    cal_pitch = (pitch + 90) % 180
    depth = calculate_depth_indices(
        image_pixel_x, image_pixel_y, depth, image_width, image_height
    )
    distance, direction = calculate_distance_and_direction(
        depth, cal_pitch, yaw, heading
    )

    return distance, direction, depth


def get_panorama_id(lat, long):
    pano = streetview.find_panorama(lat, long, radius=500)
    if pano is None:
        return None
    return str(pano.id)


def move_in_heading(lat, lon, heading, distance=0.01):
    R = 6371.0
    heading_rad = radians(heading)

    lat_rad = radians(lat)
    lon_rad = radians(lon)

    new_lat_rad = asin(
        sin(lat_rad) * cos(distance / R)
        + cos(lat_rad) * sin(distance / R) * cos(heading_rad)
    )
    new_lon_rad = lon_rad + atan2(
        sin(heading_rad) * sin(distance / R) * cos(lat_rad),
        cos(distance / R) - sin(lat_rad) * sin(new_lat_rad),
    )
    new_lat = degrees(new_lat_rad)
    new_lon = degrees(new_lon_rad)

    return new_lat, new_lon


def download_panorama_image_and_depth(pano_id):
    pano = streetview.find_panorama_by_id(pano_id, download_depth=True)
    # panorama_path = f"data/Panormas/{pano.id}_panorma.jpg"
    panorma = streetview.get_panorama(pano, zoom=5)
    streetview.download_panorama(pano, f"./panoramas/{pano.id}.jpg")
    return panorma, pano


def process_location(lat, long):
    pano_id = get_panorama_id(lat, long)
    if pano_id is None:
        return None, None, None, None, None, None
    panorma, pano = download_panorama_image_and_depth(pano_id)

    heading_degrees = degrees(pano.heading)
    if pano.depth:
        depth_map = pano.depth.data
    return panorma, depth_map, heading_degrees, pano.lat, pano.lon, pano_id
