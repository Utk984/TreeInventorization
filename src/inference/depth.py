def estimate_depth(image, model):
    depth = model.infer_image(image)
    return depth
