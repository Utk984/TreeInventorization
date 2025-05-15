import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from streetlevel import streetview
import asyncio
from config import Config
from aiohttp import ClientSession
from src.inference.segment import detect_trees
from src.inference.depth import estimate_depth
from src.utils.unwrap import divide_panorama
from src.utils.masks import add_masks, remove_duplicates, make_image
from src.utils.transformation import get_point
from src.utils.geodesic import get_coordinates
from concurrent.futures import ThreadPoolExecutor
IO_EXECUTOR = ThreadPoolExecutor(max_workers=1)  

async def fetch_pano_by_id(pano_id: str, session: ClientSession):
    pano = await streetview.find_panorama_by_id_async(
        pano_id, session, download_depth=True
    )
    rgb = await streetview.get_panorama_async(pano, session) 
    return pano, np.array(rgb)

def process_view(config: Config, view, tree_data, pano, image, depth, theta, i):
    trees = []
    for j, tree in enumerate(tree_data):
        masks = tree.masks
        boxes = tree.boxes
        if masks is not None:
            for k, mask in enumerate(masks):
                image_path = os.path.join(config.VIEW_DIR, f"{pano.id}_view{i}_tree{j}_box{k}.jpg")
                conf = boxes[k].conf.item()

                try:
                    orig_point, pers_point = get_point(mask, theta, pano, config.HEIGHT, config.WIDTH, config.FOV)
                    distance = depth[pers_point[0]][pers_point[1]]
                    lat, lon = get_coordinates(pano, orig_point, image.shape[1], distance)
                    IO_EXECUTOR.submit(
                        make_image, view, boxes[k], mask, image_path
                    )
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
                    "gsv_depth": pano.depth,
                    "predicted_depth": depth
                }
                trees.append(tree)
    return trees

async def process_panoramas(config: Config, depth_model, tree_model):
    pano_ids = pd.read_csv(config.PANORAMA_CSV)["pano_id"].tolist()
    trees_df = []     

    async with ClientSession() as session:
        fetch_task = asyncio.create_task(fetch_pano_by_id(pano_ids[0], session))

        for next_id in tqdm(pano_ids[1:], total=len(pano_ids)):
            pano, image = await fetch_task
            fetch_task = asyncio.create_task(fetch_pano_by_id(next_id, session))

            depth  = estimate_depth(image, depth_model)
            views  = divide_panorama(image, config.HEIGHT, config.WIDTH, config.FOV)

            trees = []
            for i, (view, theta) in enumerate(views):
                tree_data = detect_trees(view, tree_model, config.DEVICE)
                trees.extend(
                    process_view(config, view, tree_data, pano, image, depth, theta, i)
                )

            if not trees:
                continue

            df_part = remove_duplicates(
                pd.DataFrame(trees),
                image.shape[1], image.shape[0],
                config.HEIGHT, config.WIDTH, config.FOV
            )

            def _save_full(img=image.copy(), td=df_part.copy(), pid=pano.id):
                full = add_masks(img, td, config.HEIGHT, config.WIDTH, config.FOV)
                cv2.imwrite(os.path.join(config.FULL_DIR, f"{pid}.jpg"), full)

            IO_EXECUTOR.submit(_save_full)
            trees_df.append(df_part)
        pano, image = await fetch_task

    IO_EXECUTOR.shutdown(wait=True)

    final_df = pd.concat(trees_df, ignore_index=True)
    final_df.to_csv(config.OUTPUT_CSV, index=False, lineterminator="\n")
    print("\n✅ Pipeline finished.")