def detect_trees(view, model, device):
    results = model.predict(view, verbose=False, imgsz=(1024,1024), device=device)
    return results