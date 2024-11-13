import os

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("./treemodel/train/weights/best.pt")


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

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
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(
            self._img,
            XY[..., 0],
            XY[..., 1],
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        return persp


def map_perspective_point_to_original(
    perspective_x, perspective_y, FOV, THETA, PHI, width, height, img_shape
):
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


def create_images(image_path, FOV, phi, height, width):
    equ = Equirectangular(image_path)
    os.makedirs("perspectives", exist_ok=True)
    images_with_angles = []
    for theta in [-90, 90]:
        img = equ.GetPerspective(FOV, theta, phi, height, width)
        images_with_angles.append((img, theta))
        cv2.imwrite(f"./perspectives/perspective_image_{theta}.jpg", img)
    return images_with_angles


def detect_trees(image_path):
    original_img = cv2.imread(image_path)
    original_height, original_width = original_img.shape[:2]
    img_paths_with_angles = create_images(image_path, 90, 0, 720, 1080)

    # Predict tree locations on perspective images
    results = model.predict(
        [img for img, _ in img_paths_with_angles], conf=0.01, save=True
    )

    box_coords = []
    for result, (_, theta) in zip(results, img_paths_with_angles):
        for box in result.boxes:
            x_min, _, x_max, y_max = box.xyxy[0].tolist()
            perspective_x = (x_min + x_max) / 2
            perspective_y = y_max

            # Map perspective point to original panoramic coordinates
            original_x, original_y = map_perspective_point_to_original(
                perspective_x,
                perspective_y,
                90,
                theta,
                0,
                1080,
                720,
                (original_height, original_width),
            )

            box_coords.append((original_x, original_y, theta))

    result = model.predict(image_path, conf=0.01, save=True)

    os.system("rm -rf perspectives")

    return box_coords, original_width, original_height
