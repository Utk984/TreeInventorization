import cv2
import numpy as np
from streetlevel import streetview


def divide_panorama(pano, FOV=90):
    """
    Generate perspective views from an equirectangular image.

    Parameters:
    - image (PIL.Image): The input equirectangular image.
    - FOV (int): Field of View in degrees (default is 90).

    Returns:
    - images_with_angles (list): List of tuples (image_array, theta_angle).
    """
    height = 720
    width = 1080

    # get image from pano
    image = streetview.get_panorama(pano)

    # Convert PIL Image to OpenCV format (numpy array)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height_src, width_src, _ = img.shape

    images_with_angles = []

    for theta in range(-135, 140, 15):
        for phi in [0]:
            persp_img = get_perspective(
                img, FOV, theta, phi, height, width, width_src, height_src
            )
            images_with_angles.append((persp_img, theta, phi))

    return images_with_angles


def get_perspective(img, FOV, THETA, PHI, height, width, width_src, height_src):
    """
    Extract a perspective view from an equirectangular image.

    Parameters:
    - img (np.ndarray): Source equirectangular image.
    - FOV (int): Field of View in degrees.
    - THETA (int): Horizontal angle in degrees.
    - PHI (int): Vertical angle in degrees.
    - height (int): Height of the perspective image.
    - width (int): Width of the perspective image.
    - width_src (int): Source image width.
    - height_src (int): Source image height.

    Returns:
    - persp_img (np.ndarray): The resulting perspective image.
    """
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ],
        np.float32,
    )
    K_inv = np.linalg.inv(K)

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.stack([x, y, z], axis=-1) @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T

    lonlat = xyz2lonlat(xyz)
    XY = lonlat2XY(lonlat, width_src, height_src).astype(np.float32)
    persp_img = cv2.remap(
        img,
        XY[..., 0],
        XY[..., 1],
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    # Flip the output vertically
    # persp_img = cv2.flip(persp_img, 0)

    return persp_img


def xyz2lonlat(xyz):
    """
    Convert 3D XYZ coordinates to longitude/latitude.

    Parameters:
    - xyz (np.ndarray): 3D coordinates.

    Returns:
    - lonlat (np.ndarray): Longitude/latitude coordinates.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(x, z)
    hyp = np.sqrt(x**2 + z**2)
    lat = np.arctan2(y, hyp)
    return np.stack([lon, lat], axis=-1)


def lonlat2XY(lonlat, width_src, height_src):
    """
    Convert longitude/latitude to XY coordinates on the equirectangular image.

    Parameters:
    - lonlat (np.ndarray): Longitude/latitude coordinates.
    - width_src (int): Source image width.
    - height_src (int): Source image height.

    Returns:
    - XY (np.ndarray): XY coordinates on the equirectangular image.
    """
    lon, lat = lonlat[..., 0], lonlat[..., 1]
    x = (lon / (2 * np.pi) + 0.5) * width_src
    # y = (0.5 - lat / np.pi) * height_src
    y = (lat / np.pi + 0.5) * height_src
    return np.stack([x, y], axis=-1)
