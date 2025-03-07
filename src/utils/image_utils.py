import math

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


def map_perspective_point_to_original(x, y, theta, img_shape, FOV=90, phi=0):
    """
    Map a point from the perspective view back to the panorama.

    Parameters:
    - x (float): x-coordinate in the perspective view.
    - y (float): y-coordinate in the perspective view.
    - theta (float): Horizontal angle (theta) of the perspective view.
    - phi (float): Vertical angle (phi) of the perspective view.
    - FOV (float): Field of View in degrees.
    - width (int): Width of the perspective image.
    - height (int): Height of the perspective image.
    - width_src (int): Width of the source panorama image.
    - height_src (int): Height of the source panorama image.

    Returns:
    - (float, float): (x_panorama, y_panorama) coordinates on the panorama.
    """

    height, width = 720, 1080
    width_src, height_src = img_shape

    # Calculate perspective camera parameters
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # Create camera matrix
    K = np.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ],
        np.float32,
    )

    # Convert perspective point to normalized coordinates
    point = np.array([x, y, 1.0], dtype=np.float32)
    normalized = np.linalg.inv(K) @ point

    # Calculate rotation matrices
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
    R = R2 @ R1

    # Apply rotation to get 3D direction vector
    xyz = normalized @ R.T

    # Convert to longitude/latitude
    lon = np.arctan2(xyz[0], xyz[2])
    hyp = np.sqrt(xyz[0] ** 2 + xyz[2] ** 2)
    lat = np.arctan2(xyz[1], hyp)

    # Convert to equirectangular pixel coordinates
    eq_x = (lon / (2 * np.pi) + 0.5) * width_src
    eq_y = (lat / np.pi + 0.5) * height_src

    return (eq_x, eq_y)


def convert_to_lat_long(depth, heading, center, distance_x, distance_y):
    heading_rad = heading + (np.pi / 180) * (distance_x / 512) * 360

    # Convert heading from degrees to radians
    point_depth = depth[distance_y, distance_x]

    # Calculate the change in latitude and longitude
    d_lat = point_depth * math.cos(heading_rad) / 110574
    d_lng = (
        point_depth * math.sin(heading_rad) / 111320 * math.cos(math.radians(center[0]))
    )

    # Calculate the new latitude and longitude
    new_lat = center[0] + d_lat
    new_lng = center[1] + d_lng

    return new_lat, new_lng


def pano_depth2latlon(point, pano, image_shape):
    depth_map = pano.depth.data
    scale_factor_x = depth_map.shape[1] / image_shape[1]
    scale_factor_y = depth_map.shape[0] / image_shape[0]

    x, y = point
    mapped_point = (
        int((image_shape[1] - x) * scale_factor_x) % depth_map.shape[1],
        int(y * scale_factor_y) % depth_map.shape[0],
    )
    lat, long = convert_to_lat_long(
        depth_map, pano.heading, (pano.lat, pano.lon), mapped_point[0], mapped_point[1]
    )
    return lat, long


def image2latlon(mask, theta, pano, fov, phi):
    """
    Get the latitude and longitude of a tree from its image.
    """
    mask_points = mask.xy[0].astype(np.int32)
    bottom_y = np.max(mask_points[:, 1])
    bottom_x_coords = mask_points[mask_points[:, 1] == bottom_y][:, 0]
    bottom_center_x = int(np.mean(bottom_x_coords))

    persp_x, persp_y = bottom_center_x, bottom_y
    img_shape = (pano.image_sizes[5].x, pano.image_sizes[5].y)

    orig_point = map_perspective_point_to_original(
        persp_x, persp_y, theta, img_shape, FOV=fov, phi=phi
    )
    orig_point = tuple(map(int, orig_point))
    lat, lon = pano_depth2latlon(orig_point, pano, img_shape)

    return lat, lon, orig_point


def image2latlonall(mask, theta, pano):
    """
    Get latitude and longitude of all points in the mask
    """

    coordinates = []
    mask_points = mask.xy[0].astype(np.int32)

    for point in mask_points:
        persp_x, persp_y = point
        img_shape = (pano.image_sizes[5].x, pano.image_sizes[5].y)

        orig_point = map_perspective_point_to_original(
            persp_x, persp_y, theta, img_shape
        )
        orig_point = tuple(map(int, orig_point))
        lat, lon = pano_depth2latlon(orig_point, pano, img_shape)
        coordinates.append((lat, lon))

    return coordinates


def remove_duplicates(df, image_x, image_y, overlap_threshold=0.4):
    """
    Removes overlapping masks and masks contained within others.
    df: DataFrame with a column 'mask', where each mask is an Ultralytics mask object.
    iou_threshold: IoU threshold above which masks are considered overlapping.
    """
    masks = df["mask"].tolist()
    theta = df["theta"].tolist()
    binary_masks = []
    to_remove = set()

    for i, mask in enumerate(masks):
        try:
            if mask.xy:
                mask_points = mask.xy[0].astype(np.int32)
                original_points = []
                for point in mask_points:
                    orig_point = map_perspective_point_to_original(
                        point[0], point[1], theta[i], (image_x, image_y)
                    )
                    orig_point = tuple(map(int, orig_point))
                    original_points.append(orig_point)
                binary_mask = np.zeros((image_y, image_x), dtype=np.uint8)
                cv2.fillPoly(binary_mask, [np.array(original_points, np.int32)], 1)
                binary_masks.append(binary_mask)
            else:
                binary_masks.append(None)
        except (AttributeError, IndexError):
            binary_masks.append(None)

    for i in range(len(binary_masks)):
        if i in to_remove or binary_masks[i] is None:
            continue
        mask_i = binary_masks[i]
        area_i = np.sum(mask_i)
        for j in range(i + 1, len(binary_masks)):
            if j in to_remove or binary_masks[j] is None:
                continue
            mask_j = binary_masks[j]
            area_j = np.sum(mask_j)
            intersection = np.sum(np.logical_and(mask_i, mask_j))
            smaller_area = min(area_i, area_j)
            overlap_ratio = intersection / smaller_area
            if overlap_ratio > overlap_threshold:
                if area_i < area_j:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)
    return df.drop(index=list(to_remove)).reset_index(drop=True)


def add_masks(image, df):
    """
    Add masks to an image by overlaying Ultralytics masks.

    Args:
        image (numpy.ndarray): The original image.
        masks (list): A list of Ultralytics masks.
        points (list): A list of (x, y) coordinates for each mask.
        mask_color (tuple): BGR color for the mask overlay.
        alpha (float): Transparency factor (0 = invisible, 1 = solid).

    Returns:
        numpy.ndarray: The image with overlaid masks.
    """

    overlay = image.copy()
    img_shape = (image.shape[1], image.shape[0])
    for _, row in df.iterrows():
        mask = row["mask"]
        original_points = []
        if mask.xy:
            mask_points = mask.xy[0].astype(np.int32)
            for point in mask_points:
                orig_point = map_perspective_point_to_original(
                    point[0], point[1], row["theta"], img_shape
                )
                orig_point = tuple(map(int, orig_point))
                original_points.append(orig_point)

        cv2.polylines(
            overlay,
            [np.array(original_points, np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=5,
        )
        cv2.fillPoly(overlay, [np.array(original_points, np.int32)], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

        point = (int(row["image_x"]), int(row["image_y"]))
        cv2.circle(image, point, 25, (0, 0, 255), -1)

    return image


def batch_process(df, batch_size):
    """
    Splits a Pandas DataFrame into batches of rows.

    Args:
        df (pd.DataFrame): The DataFrame to process in batches.
        batch_size (int): The size of each batch.

    Yields:
        pd.DataFrame: A batch of rows from the DataFrame.
    """
    for i in range(0, len(df), batch_size):
        yield df.iloc[i : i + batch_size]
