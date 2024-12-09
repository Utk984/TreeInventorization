# Description: Functions for image processing and manipulation

import math
from math import asin, atan2, cos, degrees, radians, sin

import cv2
import numpy as np


def make_tree_image(view, box):
    """
    Create and return tree image from bounding box
    """

    # Extract bounding box coordinates
    xyxy = [int(coord) for coord in box.xyxy[0].tolist()]

    im = view.copy()
    cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
    cv2.putText(
        im,
        f"{box.conf.item():.2f}",
        (xyxy[0], xyxy[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    # Resize the image (make it smaller)
    im_small = cv2.resize(im, (0, 0), fx=0.3, fy=0.3)

    # crop the tree image
    im_crop = im[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]

    return im_crop, im_small


def map_perspective_point_to_original(perspective_x, perspective_y, THETA, img_shape):

    FOV = 90
    PHI = 0
    width = 1080
    height = 720

    # Step 1: Perspective point to normalized 3D coordinates
    f = 0.5 * width / np.tan(0.5 * FOV * np.pi / 180.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    # Convert perspective (x, y) to 3D point in camera coordinates
    # Add small offsets for rounding to help stability
    xyz = (
        np.array([perspective_x + 0.5, perspective_y + 0.5, 1.0], dtype=np.float32)
        @ K_inv.T
    )
    xyz = xyz / np.linalg.norm(xyz)  # Normalize to get a unit vector

    # Step 2: Apply rotation to match the original panoramic orientation
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T

    # Step 3: Convert 3D point to spherical coordinates (longitude and latitude)
    lon = np.arctan2(xyz[0], xyz[2])
    lat = np.arcsin(np.clip(xyz[1], -1.0, 1.0))  # Clip for stability

    # Step 4: Map spherical coordinates to original panoramic coordinates
    original_x = (lon / (2 * np.pi) + 0.5) * (img_shape[1] - 1)
    original_y = (lat / np.pi + 0.5) * (img_shape[0] - 1)

    return original_x, original_y


def pano_depth2latlon(orig_point, pano, theta):
    pano_size = (pano.image_sizes[5].x, pano.image_sizes[5].y)
    distance, direction, depth = calculate_3d(
        orig_point,
        abs(theta - 90),
        0,
        pano.depth.data,
        pano_size,
        pano.heading,
    )

    if depth >= 0 and distance >= 0:
        lat, lng = move_in_heading(pano.lat, pano.lon, int(direction), distance / 1000)
        return lat, lng
    return None, None


def calculate_3d(
    orig_point,
    yaw,
    pitch,
    depth,
    img_size,
    heading,
):

    image_pixel_x = orig_point[0]
    image_pixel_y = orig_point[1]

    image_height = img_size[0]
    image_width = img_size[1]

    cal_pitch = (pitch + 90) % 180
    depth = calculate_depth_indices(
        image_pixel_x, image_pixel_y, depth, image_width, image_height
    )
    distance, direction = calculate_distance_and_direction(
        depth, cal_pitch, yaw, heading
    )

    return distance, direction, depth


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


def calculate_depth_indices(
    image_pixel_x, image_pixel_y, depth, image_width, image_height
):
    index_y = int(image_pixel_y * (depth.shape[0] / image_height))
    index_x = int(image_pixel_x * (depth.shape[1] / image_width)) * (-1)
    return depth[index_y][index_x]


def calculate_image_pixel_coordinates(cal_yaw, cal_pitch, image_width, image_height):
    image_pixel_x = cal_yaw * (image_width / 360)
    image_pixel_y = cal_pitch * (image_height / 180)
    return image_pixel_x, image_pixel_y


def calculate_distance_and_direction(depth, pitch, yaw, heading):
    distance = depth * math.sin((180 - pitch) / 360)
    direction = yaw - 270 + heading
    if direction < 0:
        direction += 360
    return distance, direction
