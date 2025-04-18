import cv2
import numpy as np
from src.process.transformation import map_perspective_point_to_original
from shapely.geometry import Polygon
from shapely.strtree import STRtree

def make_image(view, box, mask, image_path):
    """
    Create and save a tree image with bounding box, confidence, and mask overlay.
    """
    xyxy = [int(coord) for coord in box.xyxy[0].tolist()]

    im = view.copy()
    cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
    cv2.putText(
        im,
        f"{box.conf.item():.2f}",
        (xyxy[0], max(xyxy[1] - 5, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )

    mask_points = mask.xy[0].astype(np.int32)
    overlay = view.copy()
    cv2.fillPoly(overlay, [mask_points], (0, 255, 0))
    cv2.addWeighted(overlay, 0.3, im, 0.7, 0, im)
    cv2.polylines(im, [mask_points], isClosed=True, color=(0, 255, 0), thickness=1)

    cv2.imwrite(image_path, im)

def remove_duplicates( df, image_x, image_y, height, width, fov, iou_threshold = 0.4):
    """
    * Converts each mask to a Shapely Polygon (no giant binary images)
    * Uses an STRtree to query only potentially overlapping pairs
    """
    polys, areas, idx_map = [], [], [] 

    for idx, (mask, th) in enumerate(zip(df["mask"], df["theta"])):
        if not mask.xy:
            continue

        pts = mask.xy[0].astype(np.float32)
        pano_pts = np.vstack(
            [
                map_perspective_point_to_original(
                    x, y, th, (image_x, image_y), height, width, fov
                )
                for x, y in pts
            ]
        )
        poly = Polygon(pano_pts).buffer(0)  
        if poly.is_empty or not poly.is_valid:
            continue

        polys.append(poly)
        areas.append(poly.area)
        idx_map.append(idx)

    if not polys:
        return df.iloc[0:0]        
    
    tree = STRtree(polys)
    keep = set(range(len(polys)))  

    for i in range(len(polys)):
        if i not in keep:
            continue
        cand = tree.query(polys[i])  
        for j in cand:
            if j <= i or j not in keep:
                continue
            inter = polys[i].intersection(polys[j]).area
            if inter == 0:
                continue
            smaller = areas[i] if areas[i] < areas[j] else areas[j]
            if inter / smaller > iou_threshold:
                keep.discard(i if areas[i] < areas[j] else j)

    keep_df_idx = [idx_map[k] for k in keep]
    return df.iloc[keep_df_idx].reset_index(drop=True)

def add_masks(image, df, height, width, FOV):
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
                    point[0], point[1], row["theta"], img_shape, height, width, FOV
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