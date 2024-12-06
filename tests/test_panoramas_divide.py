import os
import sys

import boto3
import cv2
import google.generativeai as genai
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from PIL.Image import Image
from streetlevel import streetview

from config import Config
from pipeline.image2bucket import cloud_save_image
from pipeline.image2latlon import image2latlon
from pipeline.segmentation import detect_trees
from pipeline.species_detection import get_species
from pipeline.unwrapping import divide_panorama
from utils.database import Database
from utils.image_utils import make_tree_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


matplotlib.use("TkAgg")

config = Config()
db = Database(config.DB_URL)

# panoramas = pd.read_csv(config.PANORAMA_CSV)

pano_id = "YHuImFJqNsGqB6W2tKxB8w"
pano = streetview.find_panorama_by_id(pano_id, download_depth=True)

# model init
genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
)

# AWS S3 init
s3 = boto3.client("s3", region_name="ap-south-1")

print("Processing panorama:", pano.id)
print("Found Panorama: ", pano.id)

# show the panorama
image = streetview.get_panorama(pano)

print(f"Processing panorama at {pano.lat}, {pano.lon}")

image.show()

views = divide_panorama(pano)

for i, (view, theta) in enumerate(views):
    tree_data = detect_trees(view)

    print("len(tree_data)", len(tree_data))

    for j, tree in enumerate(tree_data):
        bboxes = tree.boxes

        for k, box in enumerate(bboxes):
            # for each tree detected, overlay bbox on image and save it

            im_crop, im = make_tree_image(view, box)

            cv2.imshow("Tree", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow("Tree Crop", im_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            image_path = f"{pano.id}_view{i}_tree{j}_box{k}.jpg"
            # cloud_save_image(im, image_path, s3, config.CLOUD_STORAGE_BUCKET)

            # 5. Get lat long of tree from its image
            # lat, lon, orig_point = image2latlon(box, theta, pano)

            # 6. Get species, common name, description of tree
            # species, common_name, description = get_species(im_og, pano.address, model)

            # 7. Save annotation to database
            image_path = config.CLOUD_URL + image_path
            print(image_path)

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
            #     address="test address",
            #     elevation=pano.elevation,
            #     heading=pano.heading,
            # )


db.close()
