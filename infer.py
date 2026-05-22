from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

import cv2
import numpy as np

MODEL_PATH = Path("models/patchcore_memory.pkl")


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )

        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


def load_memory_bank():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def preprocess_image(image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def extract_patch_features(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)

        f2, f3 = model(image_tensor)

        f3 = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        features = torch.cat([f2, f3], dim=1)

        features = (
            features
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, features.shape[1])
        )

        return features.cpu()


def predict_image(image, category="hazelnut", threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    memory_data = load_memory_bank()

    if category not in memory_data:
        raise ValueError(f"Category '{category}' not found in memory bank.")

    category_data = memory_data[category]
    image_size = category_data["image_size"]
    memory_bank = category_data["memory_bank"]

    model = FeatureExtractor().to(device)
    model.eval()

    image_tensor = preprocess_image(image, image_size)
    features = extract_patch_features(model, image_tensor, device)

    # Her patch için memory bank'teki en yakın normal patch mesafesi
    distances = torch.cdist(features, memory_bank)

    min_distances, _ = torch.min(distances, dim=1)

    # Image-level anomaly score
    anomaly_score = float(torch.quantile(min_distances, 0.99).item())

    # Patch score map oluştur
    num_patches = int(min_distances.shape[0] ** 0.5)

    score_map = min_distances.reshape(num_patches, num_patches).numpy()

    # Score map'i image boyutuna büyüt
    score_map = cv2.resize(
        score_map,
        (image.size[0], image.size[1])
    )

    # Normalize et
    score_map = score_map - score_map.min()
    score_map = score_map / (score_map.max() + 1e-8)

    # Heatmap oluştur
    heatmap = np.uint8(255 * score_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Original image numpy
    original = np.array(image.convert("RGB"))

    # Overlay
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    prediction = "ANOMALY" if anomaly_score >= threshold else "NORMAL"

    return {
        "category": category,
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "heatmap": heatmap,
        "overlay": overlay,
    }