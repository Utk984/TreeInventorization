"""
Script to download all panoramas, and all 3 views with model predicition
"""

import os

import cv2
import pandas as pd
from streetlevel import streetview
from tqdm import tqdm

# from config import Config
from pipeline.segmentation import detect_trees
from pipeline.unwrapping import divide_panorama

# config
# config = Config()

# panoramas = pd.read_csv(config.PANORAMA_CSV)
panoramas = pd.read_csv("./cdg_st_v3_28_29_panoramas.csv")

print(f"Total panoramas to process: {len(panoramas)}")


for i, row in panoramas.iterrows():
    pano_id = row["pano_id"]

    try:
        pano = streetview.find_panorama_by_id(pano_id, download_depth=True)
    except Exception as e:
        print(f"Error finding panorama {pano_id}: {e}")
        continue

    # check if image exists at the path
    # if panorama already exists dont download again
    if os.path.exists(f"./data/images/verify/trees/panos/{i}_{pano_id}.jpg"):
        print(f"Panorama {i} already exists")

    else:
        print(f"\nDownloading panorama {pano_id}...")
        streetview.download_panorama(
            pano, f"./data/images/verify/trees/panos/{i}_{pano_id}.jpg"
        )
        print(f"Saved panorama {i}")

    print(f"\nProcessing panorama {pano_id}...")

    # 1. Divide panorama into 90-degree views
    views = divide_panorama(pano)

    for j, (view, theta) in enumerate(views):

        # 2. Run instance segmentation model to detect trees
        tree_data = detect_trees(view)

        # if view already exists dont download again
        if os.path.exists(f"./data/images/verify/trees/views/{i}_view{j}.png"):
            print(f"View {j} for panorama {i} already exists")
            continue

        # make an image with the view and all its boxes
        im = view.copy()
        for tree in tree_data:
            bboxes = tree.boxes
            for box in bboxes:
                xyxy = [int(coord) for coord in box.xyxy[0].tolist()]
                cv2.rectangle(
                    im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2
                )
                cv2.putText(
                    im,
                    f"{box.conf.item():.2f}",
                    (xyxy[0], xyxy[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                # im = box.draw_on_image(im)

        # save the image
        # it is a numpy ndarray
        cv2.imwrite(f"./data/images/verify/trees/views/{i}_view{j}.png", im)
        # im.save(f"./data/images/verify/views/no{i}_view{j}.png")
        print(f"Saved view {j} for panorama {i}")
        cv2.destroyAllWindows()
