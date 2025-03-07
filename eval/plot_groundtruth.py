import os

import cv2
import pandas as pd

image_dir = "../data/images/full/"
save_dir = "../data/images/test/"

data = pd.read_csv("./28_29_groundtruth.csv")

for index, row in data.iterrows():
    pano_id = row["pano_id"]
    image_x = row["image_x"]
    image_y = row["image_y"]
    image_path = os.path.join(image_dir, f"{pano_id}_full.jpg")

    if os.path.exists(image_path):
        img = cv2.imread(image_path)

        if img is not None:
            # Convert floating-point coordinates to integers
            x, y = int(image_x), int(image_y)

            # Draw a red circle on the image
            cv2.circle(img, (x, y), radius=35, color=(255, 0, 0), thickness=-1)

            # Save the modified image
            output_path = os.path.join(save_dir, f"{pano_id}_marked.jpg")
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to load image: {image_path}")
    else:
        print(f"Image not found: {image_path}")
