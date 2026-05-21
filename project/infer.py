def predict_image(image, threshold=0.5):

    # Dummy anomaly score
    anomaly_score = 0.42

    # Threshold kontrolü
    if anomaly_score >= threshold:
        prediction = "ANOMALY"
    else:
        prediction = "NORMAL"

    return {
        "prediction": prediction,
        "anomaly_score": anomaly_score
    }