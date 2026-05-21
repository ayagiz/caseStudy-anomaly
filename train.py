from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


CATEGORIES = ["hazelnut", "wood"]
DATA_ROOT = Path("data/")
SAVE_PATH = Path("models/patchcore_memory.pkl")
IMAGE_SIZE = 224


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.Wide_ResNet50_2_Weights.DEFAULT
        backbone = models.wide_resnet50_2(weights=weights)

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


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
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


def build_memory_bank_for_category(model, category, device):
    train_path = DATA_ROOT / category / "train" / "good"
    image_paths = sorted(list(train_path.glob("*.png")))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No training images found in: {train_path}")

    print(f"\nCategory: {category}")
    print(f"Found {len(image_paths)} training images.")

    memory_bank = []

    for idx, image_path in enumerate(image_paths):
        image_tensor = preprocess_image(image_path)
        features = extract_patch_features(model, image_tensor, device)
        memory_bank.append(features)

        print(f"[{idx + 1}/{len(image_paths)}] {image_path.name}")

    memory_bank = torch.cat(memory_bank, dim=0)

    print(f"{category} memory bank shape: {memory_bank.shape}")

    return memory_bank


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = FeatureExtractor().to(device)
    model.eval()

    all_memory_banks = {}

    for category in CATEGORIES:
        memory_bank = build_memory_bank_for_category(
            model=model,
            category=category,
            device=device,
        )

        all_memory_banks[category] = {
            "image_size": IMAGE_SIZE,
            "memory_bank": memory_bank,
        }

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(all_memory_banks, f)

    print(f"\nSaved all memory banks to: {SAVE_PATH}")


if __name__ == "__main__":
    main()