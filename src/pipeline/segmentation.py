from ultralytics import YOLO

model = YOLO("./treemodel/train/weights/best.pt")


def detect_trees(view):
    """
    Detect trees in the panorama, using the instance segmentation model build on YOLO
    """

    results = model.predict(view, conf=0.1, verbose=False)

    return results
