"""
Extract tree latlon from image
"""

from utils.image_utils import map_perspective_point_to_original, pano_depth2latlon


def image2latlon(box, theta, pano):
    """
    Get the latitude and longitude of a tree from its image.
    """

    xyxy = [int(coord) for coord in box.xyxy[0].tolist()]
    persp_x = (xyxy[0] + xyxy[2]) / 2
    persp_y = xyxy[3]

    img_shape = (pano.image_sizes[5].x, pano.image_sizes[5].y)

    # orig_point holds a point on the pano image, where the tree is
    orig_point = map_perspective_point_to_original(persp_x, persp_y, theta, img_shape)

    lat, lon = pano_depth2latlon(orig_point, pano, theta)

    return lat, lon, orig_point
