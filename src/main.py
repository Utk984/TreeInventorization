"""
src/main.py
Main entry point for the urban tree inventory pipeline.
"""

import boto3
import google.generativeai as genai
import pandas as pd
from openai import OpenAI
from streetlevel import streetview
from tqdm import tqdm

from config import Config
from pipeline.cloud_storage import cloud_save_image, local_save_image
from pipeline.database import Database, save_to_csv
from pipeline.segmentation import detect_trees
from pipeline.species_detection import get_species_gemini, get_species_gpt
from pipeline.tree_extraction import image2latlon
from pipeline.unwrapping import divide_panorama
from utils.batch_processing import batch_process
from utils.image_utils import make_tree_image


def process_panorama_batch(config):
    """
    Process panoramas in batches.
    """
    print("Urban Tree Inventory Pipeline started.")
    print("Loading panoramas from CSV...")

    # Load panoramas from CSV
    panoramas = pd.read_csv(config.PANORAMA_CSV)

    start_row = 0

    # use panoramas df from start row onwards
    # panoramas = panoramas.iloc[start_row:]
    print(f"Total panoramas to process: {len(panoramas)}")

    # Model and AWS S3 initialization
    print("Initializing model and AWS S3 client...")
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")
    client = OpenAI()

    # s3 = boto3.client("s3", region_name="ap-south-1")

    # Initialize counters
    total_trees = 0
    total_batches = 0
    total_panos_processed = 0

    # Batch process panoramas
    for batch_idx, batch in enumerate(
        batch_process(panoramas, config.BATCH_SIZE), start=1
    ):
        print(f"\nProcessing batch {batch_idx}...")

        # Connect to the database
        # db = Database(config.DB_URL)
        # print("Connecting to the database...")

        batch_trees = 0

        # for _, row in tqdm(batch.iterrows(), total=len(batch)):
        for _, row in batch.iterrows():
            pano_id = row["pano_id"]

            try:
                pano = streetview.find_panorama_by_id(pano_id, download_depth=True)
            except Exception as e:
                print(f"Error finding panorama {pano_id}: {e}")
                continue

            print(f"\nProcessing panorama {pano_id}...")

            # 1. Divide panorama into 90-degree views
            views = divide_panorama(pano)

            for i, (view, theta) in enumerate(views):

                # 2. Run instance segmentation model to detect trees
                tree_data = detect_trees(view)

                for j, tree in enumerate(tree_data):
                    bboxes = tree.boxes

                    for k, box in enumerate(bboxes):
                        try:
                            # 3. Get tree image from bounding box
                            im_crop, im = make_tree_image(view, box)

                            # 4. Save tree image to cloud storage bucket
                            image_path = f"{pano.id}_view{i}_tree{j}_box{k}.jpg"
                            # cloud_save_image(
                            #     im, image_path, s3, config.CLOUD_STORAGE_BUCKET
                            # )
                            local_save_image(im, config.OUTPUT_DIR, image_path)

                            # 5. Get lat long of tree from its image
                            lat, lon, orig_point = image2latlon(box, theta, pano)

                            # 6. Get species, common name, description of tree
                            gem_species, gem_common_name, gem_description = (
                                get_species_gemini(im_crop, pano.address, model)
                            )
                            gpt_species, gpt_common_name, gpt_description = (
                                get_species_gpt(im_crop, pano.address, client)
                            )

                            # 7. Save annotation to database (OR CSV)
                            image_path = config.CLOUD_URL + image_path
                            tree = {
                                "image_path": image_path,
                                "pano_id": pano_id,
                                "stview_lat": pano.lat,
                                "stview_lng": pano.lon,
                                "tree_lat": lat,
                                "tree_lng": lon,
                                "lat_offset": 0,
                                "lng_offset": 0,
                                "image_x": float(orig_point[0]),
                                "image_y": float(orig_point[1]),
                                "height": 0,
                                "diameter": 0,
                                "gem_species": gem_species,
                                "gem_common_name": gem_common_name,
                                "gem_description": gem_description,
                                "gpt_species": gpt_species,
                                "gpt_common_name": gpt_common_name,
                                "gpt_description": gpt_description,
                                "theta": theta,
                                "address": row["address"],
                                "elevation": row["elevation"],
                                "heading": row["heading"],
                            }
                            save_to_csv(tree, config.STREET_OUTPUT_CSV)
                            # db.insert_annotation(
                            #     image_path=image_path,
                            #     pano_id=pano_id,
                            #     stview_lat=pano.lat,
                            #     stview_lng=pano.lon,
                            #     tree_lat=lat,
                            #     tree_lng=lon,
                            #     lat_offset=0,
                            #     lng_offset=0,
                            #     image_x=float(orig_point[0]),
                            #     image_y=float(orig_point[1]),
                            #     height=0,
                            #     diameter=0,
                            #     species=species,
                            #     common_name=common_name,
                            #     description=description,
                            #     theta=theta,
                            #     address=row["address"],
                            #     elevation=row["elevation"],
                            #     heading=row["heading"],
                            # )

                            total_trees += 1
                            batch_trees += 1

                        except Exception as e:
                            print(
                                f"{pano_id}: Error processing tree {k} in view {i}: {e}"
                            )
                            continue

            total_panos_processed += 1

        print(
            f"Batch {batch_idx} processed: {len(batch)} panoramas, {batch_trees} trees detected."
        )
        total_batches += 1
        # db.close()

    print("\nPipeline finished.")
    print(f"Total batches processed: {total_batches}")
    print(f"Total panoramas processed: {total_panos_processed}")
    print(f"Total trees detected: {total_trees}")


def main():
    # Load configuration
    config = Config()

    # Process panoramas in batches
    process_panorama_batch(config)


if __name__ == "__main__":
    main()
