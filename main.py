import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import torch
from streetlevel import streetview
import time
from config import Config
from models.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2
from src.inference.segment import detect_trees
from src.inference.depth import estimate_depth
from src.unwrapping.unwrap import divide_panorama
from src.process.masks import add_masks, remove_duplicates, make_image
from src.process.transformation import get_point
from src.process.geodesic import get_coordinates


def load_models(config: Config):
    # Load depth model
    depth_model = DepthAnythingV2(**{**config.MODEL_CONFIGS["vitl"], "max_depth": 80})
    depth_model.load_state_dict(torch.load(config.DEPTH_MODEL_PATH, map_location="cpu"))
    depth_model.to(config.DEVICE).eval()

    # Load tree segmentation model
    tree_model = YOLO(config.TREE_MODEL_PATH)
    tree_model.to(config.DEVICE).eval()

    return depth_model, tree_model


def process_view(view, tree_data, pano, image, depth, theta, i):
    trees = []
    for j, tree in enumerate(tree_data):
        masks = tree.masks
        boxes = tree.boxes
        if masks is not None:
            for k, mask in enumerate(masks):
                image_path = os.path.join(config.VIEW_DIR, f"{pano.id}_view{i}_tree{j}_box{k}.jpg")
                conf = boxes[k].conf.item()

                try:
                    orig_point, pers_point = get_point(mask, theta, pano, config.FOV)
                    distance = depth[pers_point[0]][pers_point[1]]
                    lat, lon = get_coordinates(pano, orig_point, image.shape[1], distance)
                    make_image(view, boxes[k], mask, image_path)
                except Exception as e:
                    print(f"Error processing tree {k} in view {i}: {e}")
                    continue

                tree = {
                    "image_path": image_path,
                    "pano_id": pano.id,
                    "stview_lat": pano.lat,
                    "stview_lng": pano.lon,
                    "tree_lat": lat,
                    "tree_lng": lon,
                    "image_x": float(orig_point[0]),
                    "image_y": float(orig_point[1]),
                    "theta": theta,
                    "mask": mask,
                    "conf": conf,
                }
                trees.append(tree)
    return trees



def process_panorama_batch(config: Config):
    panoramas = pd.read_csv(config.PANORAMA_CSV)
    tree_df = pd.DataFrame()
    tqdm.pandas()

    depth_model, tree_model = load_models(config)

    for _, row in tqdm(panoramas.iterrows(), total=len(panoramas)):
        t1 = time.time()
        pano_id = row["pano_id"]
        trees = []

        try:
            pano = streetview.find_panorama_by_id(pano_id, download_depth=False)
            t2 = time.time()
            print("Getting pano ", t2-t1)
            image = np.array(streetview.get_panorama(pano))
            t3 = time.time()
            print("Getting image ", t3-t2)
        except Exception as e:
            print(f"Error finding panorama {pano_id}: {e}")
            continue
        
        depth = estimate_depth(image, depth_model)
        t4 = time.time()
        print("Getting depth ", t4-t3)
        views = divide_panorama(image, FOV=config.FOV)
        t5 = time.time()
        print("Getting views ", t5-t4)

        for i, (view, theta) in enumerate(views):
            tree_data = detect_trees(view, tree_model, config.DEVICE)
            trees.extend(process_view(view, tree_data, pano, image, depth, theta, i))
    
        if not trees:
            continue

        t6 = time.time()
        print("Getting trees ", t6-t5)

        tree_data = remove_duplicates(pd.DataFrame(trees), image.shape[1], image.shape[0])
        image = add_masks(image, tree_data, config.FOV)
        cv2.imwrite(os.path.join(config.FULL_DIR, f"{pano.id}.jpg"), image)
        tree_df = pd.concat([tree_df, tree_data], ignore_index=True)

        t7 = time.time()
        print("Getting df ", t7-t6)

    print("\n✅ Pipeline finished.")
    tree_df.to_csv(config.STREET_OUTPUT_CSV, index=False, lineterminator="\n")


if __name__ == "__main__":
    config = Config()
    process_panorama_batch(config)