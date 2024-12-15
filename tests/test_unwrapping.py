import os
import sys

import cv2
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from PIL.Image import Image
from streetlevel import streetview

from pipeline.segmentation import detect_trees
from pipeline.unwrapping import divide_panorama
from utils.image_utils import make_tree_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

matplotlib.use("TkAgg")


# pano_id = "YHuImFJqNsGqB6W2tKxB8w"
pano_id = "wisoo7e5XcA1arbsV6N7Lw"
pano = streetview.find_panorama_by_id(pano_id, download_depth=True)

image = streetview.get_panorama(pano)

# resize image
image = image.resize(
    (int(image.width / 3), int(image.height / 3))
)  # , Image.ANTIALIAS)


# save PIL image
# image.save(f"data/images/panoramas/{pano_id}.jpg")

views = divide_panorama(pano)

for i, (view, theta) in enumerate(views):
    # resize image
    view = cv2.resize(
        view, (int(view.shape[1] / 2), int(view.shape[0] / 2))
    )  # , Image.ANTIALIAS)

    # save numpy array as image
    cv2.imwrite(f"data/images/perspectives/{pano_id}_view{i}.jpg", view)

    tree_data = detect_trees(view)

    print("len(tree_data)", len(tree_data))

    for j, tree in enumerate(tree_data):
        bboxes = tree.boxes
        print("len(bboxes)", len(bboxes))

        for k, box in enumerate(bboxes):
            # for each tree detected, overlay bbox on image and save it
            # print("box: ", box)

            im_crop, im = make_tree_image(view, box)

            print("im_crop: ", im_crop.shape)
            print("im: ", im.shape)

            # plot both in 1 window
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(im)
            ax[1].imshow(im_crop)
            plt.show()

            image_path = f"{pano.id}_view{i}_tree{j}_box{k}.jpg"
            print("image_path: ", image_path)
