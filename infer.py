def predict_image(image, category="hazelnut", threshold=0.5):
    anomaly_score = 0.42

    prediction = "ANOMALY" if anomaly_score >= threshold else "NORMAL"

    return {
        "category": category,
        "prediction": prediction,
        "anomaly_score": anomaly_score
    }