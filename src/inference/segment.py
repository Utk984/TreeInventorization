def detect_trees(view, model):
    results = model.predict(view, verbose=False)
    return results