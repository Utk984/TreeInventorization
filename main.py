import os

import cv2
import numpy as np
import pandas as pd
from streetlevel import streetview
from tqdm import tqdm

from src.pipeline.segmentation import detect_trees
from src.pipeline.unwrapping import divide_panorama
from src.utils.image_utils import (add_masks, image2latlon, image2latlonall,
                                   remove_duplicates)


def process_panorama_batch(fov=90):
    """
    Process panoramas in batches.
    """
    panoramas = pd.read_csv("./streets/south_delhi_st2_vinaymarg_panoramas.csv")
    panoramas2 = pd.read_csv("./streets/south_delhi_st1_satyamarg_panoramas.csv")
    panoramas = pd.concat([panoramas, panoramas2], ignore_index=True)

    print("Urban Tree Inventory Pipeline started.")
    print(f"Total panoramas to process: {len(panoramas)}")

    tree_df = pd.DataFrame()
    tqdm.pandas()

    for _, row in tqdm(panoramas.iterrows()):
        pano_id = row["pano_id"]
        trees = []
        try:
            pano = streetview.find_panorama_by_id(pano_id, download_depth=True)
        except Exception as e:
            print(f"Error finding panorama {pano_id}: {e}")
            continue

        views = divide_panorama(pano, FOV=fov)
        image = np.array(streetview.get_panorama(pano))

        for i, (view, theta, phi) in enumerate(views):
            tree_data = detect_trees(view)
            for j, tree in enumerate(tree_data):
                masks = tree.masks
                boxes = tree.boxes
                if masks is not None:
                    for k, mask in enumerate(masks):
                        conf = boxes[k].conf.item()
                        try:
                            lat, lon, orig_point = image2latlon(
                                mask, theta, pano, fov, phi
                            )
                            # coordinates = image2latlonall(mask, theta, pano)
                        except Exception as e:
                            print(f"Error processing tree {k} in view {i}: {e}")
                            continue

                        if lat is None or lon is None:
                            print(f"Tree {k} in view {i} not found.")
                            continue

                        image_path = (
                            f"./data/images/views/{pano.id}_view{i}_tree{j}_box{k}.jpg"
                        )
                        mask_points = mask.xy[0].astype(np.int32)
                        overlay = view.copy()
                        cv2.fillPoly(overlay, np.int32([mask_points]), (0, 255, 0))
                        cv2.addWeighted(overlay, 0.3, view, 0.7, 0, view)
                        cv2.polylines(
                            view,
                            np.int32([mask_points]),
                            isClosed=True,
                            color=(0, 255, 0),
                            thickness=1,
                        )
                        cv2.imwrite(image_path, view)

                        tree = {
                            "image_path": f"./data/images/tests/{pano.id}_view{i}_tree{j}_box{k}.jpg",
                            "pano_id": pano_id,
                            "stview_lat": pano.lat,
                            "stview_lng": pano.lon,
                            "tree_lat": lat,
                            "tree_lng": lon,
                            "lat_offset": 0,
                            "lng_offset": 0,
                            "image_x": float(orig_point[0]),
                            "image_y": float(orig_point[1]),
                            "theta": theta,
                            "mask": mask,
                            "conf": conf,
                            # "coordinates": coordinates,
                        }
                        trees.append(tree)

        if len(trees) == 0:
            continue
        # tree_data = remove_duplicates(
        #     pd.DataFrame(trees), image.shape[1], image.shape[0]
        # )
        tree_data = pd.DataFrame(trees)
        image = add_masks(image, tree_data)
        cv2.imwrite(f"./data/images/full/{pano.id}.jpg", image)
        tree_df = pd.concat([tree_df, tree_data], ignore_index=True)

    print("\nPipeline finished.")
    tree_df.to_csv("tree_data.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    fov = 90
    process_panorama_batch(fov)
