import cv2
import numpy as np
from src.process.transformation import map_perspective_point_to_original
import torch
from torchvision.ops import nms

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

def remove_duplicates2(df, image_x, image_y, iou_threshold=0.4):
    """
    Performs bounding box NMS using transformed mask polygons in the full panorama.
    """
    masks = df["mask"].tolist()
    theta = df["theta"].tolist()

    boxes = []
    scores = []
    valid_indices = []

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

                # Create full-pano binary mask
                binary_mask = np.zeros((image_y, image_x), dtype=np.uint8)
                cv2.fillPoly(binary_mask, [np.array(original_points, np.int32)], 1)

                # Get bounding box on the pano-aligned mask
                x, y, w, h = cv2.boundingRect(np.array(original_points))
                box = [x, y, x + w, y + h]  # [x1, y1, x2, y2]

                boxes.append(box)
                scores.append(df["conf"].iloc[i])
                valid_indices.append(i)
        except Exception as e:
            continue

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    keep = nms(boxes_tensor, scores_tensor, iou_threshold)
    kept_indices = [valid_indices[i] for i in keep.numpy()]
    return df.iloc[kept_indices].reset_index(drop=True)

def remove_duplicates(df, image_x, image_y, height, width, FOV, iou_threshold=0.4):
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
                        point[0], point[1], theta[i], (image_x, image_y), height, width, FOV
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
            if overlap_ratio > iou_threshold:
                if area_i < area_j:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)
    
    return df.drop(index=list(to_remove)).reset_index(drop=True)


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