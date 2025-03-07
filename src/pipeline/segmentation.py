from ultralytics import YOLO

model = YOLO("./treemodel/train/weights/best.pt")


def detect_trees(view):
    results = model.predict(view, verbose=False)
    return results
