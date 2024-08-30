def aggregate_predictions(predictions, threshold=0.5):
    fake_count = sum([1 for p in predictions if p > threshold])
    real_count = len(predictions) - fake_count
    
    if fake_count > real_count:
        return "Deepfake detected!"
    else:
        return "Real video detected."
