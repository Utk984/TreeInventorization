import cv2
import numpy as np

def map_perspective_point_to_original(x, y, theta, img_shape, height, width, FOV):
    PHI = 0
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
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
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


def get_point(mask, theta, pano, height, width, FOV):
    mask_points = mask.xy[0].astype(np.int32)
    bottom_y = np.max(mask_points[:, 1])
    bottom_x_coords = mask_points[mask_points[:, 1] == bottom_y][:, 0]
    bottom_center_x = int(np.mean(bottom_x_coords))

    img_shape = (pano.image_sizes[5].x, pano.image_sizes[5].y)

    orig_point = map_perspective_point_to_original(
        bottom_center_x, bottom_y, theta, img_shape, height, width, FOV
    )
    orig_point = tuple(map(int, orig_point))
    return orig_point, (bottom_center_x, bottom_y)