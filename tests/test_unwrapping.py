import os
import sys

import cv2
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from PIL.Image import Image
from streetlevel import streetview

from pipeline.unwrapping import divide_panorama

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

matplotlib.use("TkAgg")


pano_id = "YHuImFJqNsGqB6W2tKxB8w"
pano = streetview.find_panorama_by_id(pano_id, download_depth=True)

image = streetview.get_panorama(pano)

# resize image
image = image.resize((int(image.width / 3), int(image.height / 3)))# , Image.ANTIALIAS)


# save PIL image
image.save(f"data/images/panoramas/{pano_id}.jpg")

views = divide_panorama(pano)

for i, (view, theta) in enumerate(views):
    # resize image
    view = cv2.resize(view, (int(view.shape[1] / 2), int(view.shape[0] / 2)))  # , Image.ANTIALIAS)

    # save numpy array as image
    cv2.imwrite(f"data/images/perspectives/{pano_id}_view{i}.jpg", view)
