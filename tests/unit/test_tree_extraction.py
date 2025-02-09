"""
Test for tree extraction
1. take a panorama
2. divide into views
3. detect trees in each view
4. find points on all trees, and plot those points on view
5. project all view's points onto original panorama
6. plot all points on original panorama
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from streetlevel import streetview

from pipeline.segmentation import detect_trees
from pipeline.tree_extraction import image2latlon
from pipeline.unwrapping import divide_panorama, lonlat2XY, xyz2lonlat


def perspective_to_panorama(x, y, theta, phi, FOV, width, height, width_src, height_src):
    """
    Map a point from the perspective view back to the panorama.

    Parameters:
    - x (float): x-coordinate in the perspective view.
    - y (float): y-coordinate in the perspective view.
    - theta (float): Horizontal angle (theta) of the perspective view.
    - phi (float): Vertical angle (phi) of the perspective view.
    - FOV (float): Field of View in degrees.
    - width (int): Width of the perspective image.
    - height (int): Height of the perspective image.
    - width_src (int): Width of the source panorama image.
    - height_src (int): Height of the source panorama image.

    Returns:
    - (float, float): (x_panorama, y_panorama) coordinates on the panorama.
    """
    # Step 1: Compute intrinsic matrix K for the perspective view
    f = 0.5 * width / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    # Step 2: Convert pixel (x, y) to normalized device coordinates (NDC)
    p_ndc = np.array([x, y, 1.0], dtype=np.float32) @ K_inv.T  # (3,)

    # Step 3: Apply reverse rotation
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(-y_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(-np.dot(R1, x_axis) * np.radians(phi))
    R = R1 @ R2
    p_world = p_ndc @ R.T  # Reverse rotation

    # Step 4: Convert 3D world coordinates to longitude/latitude
    lonlat = xyz2lonlat(p_world.reshape(-1, 3))

    # Step 5: Map longitude/latitude to panorama coordinates
    XY = lonlat2XY(lonlat, width_src, height_src)

    # Return the mapped panorama coordinates
    return XY[0, 0], XY[0, 1]


def test_tree_extraction():
    # 1. take a panorama
    pano_id = "MEp5WWE7sF_STLkpOhpzdA"
    # pano_id = "DpP0k335r-GwRuatCL5mSQ"
    pano = streetview.find_panorama_by_id(pano_id, download_depth=True)

    csv_path = "./data/output/street_panoramas_final.csv"
    df = pd.read_csv(csv_path)
    pano_rows = df[df["pano_id"] == pano_id]

    if pano is None:
        print("Panorama not found")
        return

    pano_img = streetview.get_panorama(pano)

    orig_points = []
    orig_bboxes = []

    # 2. divide into viewsj
    views = divide_panorama(pano)

    for i, (view, theta) in enumerate(views):
        # 3. detect trees in each view
        tree_data = detect_trees(view)

        im = view.copy()
        for j, tree in enumerate(tree_data):
            bboxes = tree.boxes

            for k, box in enumerate(bboxes):

                # 4. find points on all trees, and plot those points on view
                xyxy = [int(coord) for coord in box.xyxy[0].tolist()]
                cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(
                    im,
                    f"{box.conf.item():.2f}",
                    (xyxy[0], xyxy[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                persp_x = (xyxy[0] + xyxy[2]) / 2
                persp_y = xyxy[3]
                print(persp_x, persp_y)
                cv2.circle(im, (int(persp_x), int(persp_y)), 5, (255, 0, 0), -1)

                # 5. project all view's points onto original panorama
                # _, _, orig_point = image2latlon(box, theta, pano)
                orig_point = perspective_to_panorama(
                    persp_x,
                    persp_y,
                    theta,
                    0,
                    90,
                    1080,
                    720,
                    pano.image_sizes[5].x,
                    pano.image_sizes[5].y,
                )
                orig_points.append(orig_point)

                # convert xyxy to original panorama coordinates
                orig_bbox_1 = perspective_to_panorama(
                    xyxy[0],
                    xyxy[1],
                    theta,
                    0,
                    90,
                    1080,
                    720,
                    pano.image_sizes[5].x,
                    pano.image_sizes[5].y,
                )
                orig_bbox_2 = perspective_to_panorama(
                    xyxy[2],
                    xyxy[3],
                    theta,
                    0,
                    90,
                    1080,
                    720,
                    pano.image_sizes[5].x,
                    pano.image_sizes[5].y,
                )
                orig_bboxes.append((orig_bbox_1, orig_bbox_2))

        # save the image
        cv2.imwrite(f"./data/images/test/view{i}.png", im)
        plt.imshow(im)

    # 6. plot all points on original panorama
    # use PIL
    pano_im = cv2.cvtColor(np.array(pano_img), cv2.COLOR_RGB2BGR)
    for point in orig_points:
        p = (int(point[0]), int(point[1]))
        print(point)
        cv2.circle(pano_im, p, 50, (0, 0, 255), -1)

    for bbox in orig_bboxes:
        cv2.rectangle(pano_im, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), (0, 0, 255), 2)

    # resize the image
    pano_im = cv2.resize(pano_im, (0, 0), fx=0.3, fy=0.3)
    cv2.imwrite("./data/images/test/pano.png", pano_im)

    # print 2 columns from pano_rows
    print(pano_rows[["pano_id", "tree_lat", "tree_lng", "image_x", "image_y"]])

if __name__=="__main__":
    test_tree_extraction()

