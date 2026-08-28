import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as T
from sklearn.metrics import f1_score
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_eurosat(image_size=64, normalize=False):
    tfms = [T.Resize((image_size, image_size)), T.ToTensor()]
    if normalize:
        tfms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
    dataset = torchvision.datasets.EuroSAT(
        root="./data",
        download=True,
        transform=T.Compose(tfms),
    )
    return dataset

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

MODEL_CONFIG = {
    "small_cnn":  {"image_size": 64,  "normalize": False, "pretrained": False},
    "mobilenet":  {"image_size": 224, "normalize": True,  "pretrained": True},
    "vit":        {"image_size": 224, "normalize": True,  "pretrained": True},
}

class DegradeTransform:
    """Applies sensor-noise-like degradation to a tensor image.
    Assumes input is still in [0,1] range - apply BEFORE normalization."""

    def __init__(self, level="none"):
        assert level in ("none", "moderate", "severe")
        self.level = level

    def __call__(self, img):
        if self.level == "none":
            return img

        if self.level == "moderate":
            noise_std = 0.03
            bit_depth = 6
        else:  # severe
            noise_std = 0.08
            bit_depth = 4

        img = img + torch.randn_like(img) * noise_std
        img = torch.clamp(img, 0, 1)

        levels = 2 ** bit_depth
        img = torch.round(img * (levels - 1)) / (levels - 1)

        # TODO: add "skip atmospheric correction" effect - see paper #7
        # for how they modeled this on real PAN/XS products.

        return img


def make_degraded_dataset(base_dataset, level, normalize_after=None):
    class Wrapped(Dataset):
        def __init__(self, base, level, normalize_after):
            self.base = base
            self.degrade = DegradeTransform(level)
            self.normalize_after = normalize_after
            if normalize_after:
                self.norm = T.Normalize(*normalize_after)

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            img = self.degrade(img)
            if self.normalize_after:
                img = self.norm(img)
            return img, label

    return Wrapped(base_dataset, level, normalize_after)

def make_few_shot_subset(dataset, n_per_class, seed):
    """Returns a Subset with exactly n_per_class examples per class.
    Use n_per_class=None for the full dataset (seed unused in that case)."""
    if n_per_class is None:
        return dataset

    rng = random.Random(seed)
    by_class = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        by_class.setdefault(label, []).append(idx)

    selected = []
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        selected.extend(idxs[:n_per_class])

    return Subset(dataset, selected)

class SimpleCNN(nn.Module):
    """Expects 64x64 input."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(name, num_classes=10):
    if name == "small_cnn":
        return SimpleCNN(num_classes=num_classes)
    elif name == "mobilenet":
        import timm
        return timm.create_model("mobilenetv2_100", pretrained=True,
                                  num_classes=num_classes)
    elif name == "vit":
        import timm
        return timm.create_model("vit_tiny_patch16_224", pretrained=True,
                                  num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


def get_lr(model_name):
    """From-scratch model uses standard training LR; pretrained models use a
    lower fine-tuning LR to avoid disrupting pretrained features (fixes the
    LR confound flagged in review)."""
    return 1e-3 if model_name == "small_cnn" else 1e-4


def epochs_for_shot_count(n_shot):
    if n_shot in (5, 10):
        return 50
    elif n_shot == 50:
        return 30
    else:
        return 15


def train_model(model, train_loader, epochs, lr):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters())

    start = time.time()
    print_every = 1 if epochs <= 15 else max(1, epochs // 10)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            elapsed = time.time() - start
            print(f"    epoch {epoch+1}/{epochs}  avg_loss={epoch_loss/n_batches:.4f}  "
                  f"elapsed={elapsed:.0f}s")
    train_time = time.time() - start
    return model, train_time, num_params


def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, f1


def train_and_eval(model, train_loader, test_loader, epochs, lr=1e-3):
    model, train_time, num_params = train_model(model, train_loader, epochs, lr)
    accuracy, f1 = evaluate(model, test_loader)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "train_time_sec": round(train_time, 1),
        "num_params": num_params,
    }

FINAL_MODELS = ["mobilenet", "vit"]
FINAL_SHOTS = [5, 10, 50, None]  # None = full dataset
FINAL_DEGRADATION = ["none", "moderate", "severe"]
FINAL_SEEDS = [42, 123, 7, 2024, 99, 17, 256, 512, 88, 314]  # 10 seeds, matches small_cnn

FINAL_CSV = "/content/drive/MyDrive/results_mobilenet_vit_final.csv"


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_final_rerun():
    print("=" * 60)
    print("FINAL CLEAN RERUN: mobilenet + vit @ lr=1e-4, all conditions, 10 seeds")
    print("=" * 60)

    fieldnames = ["model", "shots", "degradation", "seed", "lr",
                  "accuracy", "f1", "train_time_sec", "num_params"]
    write_header = not os.path.exists(FINAL_CSV)
    f = open(FINAL_CSV, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        f.flush()

    already_done = set()
    if os.path.exists(FINAL_CSV):
        with open(FINAL_CSV, "r") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                already_done.add((row["model"], row["shots"], row["degradation"], row["seed"]))

    for model_name in FINAL_MODELS:
        cfg = MODEL_CONFIG[model_name]
        base_dataset = load_eurosat(image_size=cfg["image_size"], normalize=False)
        train_size = int(0.8 * len(base_dataset))
        test_size = len(base_dataset) - train_size
        train_base, test_base = random_split(
            base_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(0)
        )
        normalize_after = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if cfg["normalize"] else None
        lr = get_lr(model_name)  # 1e-4 for both mobilenet and vit

        for n_shot in FINAL_SHOTS:
            for degradation in FINAL_DEGRADATION:
                for seed in FINAL_SEEDS:
                    shots_label = n_shot if n_shot else "full"
                    key = (model_name, str(shots_label), degradation, str(seed))
                    if key in already_done:
                        print(f"Skipping (already done): {key}")
                        continue

                    print(f"model={model_name}, shots={shots_label}, "
                          f"degradation={degradation}, seed={seed}, lr={lr}")
                    set_all_seeds(seed)

                    train_degraded = make_degraded_dataset(train_base, degradation, normalize_after)
                    test_degraded = make_degraded_dataset(test_base, degradation, normalize_after)
                    train_subset = make_few_shot_subset(train_degraded, n_shot, seed)

                    loader_kwargs = dict(num_workers=2, pin_memory=(DEVICE == "cuda"), persistent_workers=True)
                    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, **loader_kwargs)
                    test_loader = DataLoader(test_degraded, batch_size=64, **loader_kwargs)

                    model = get_model(model_name)
                    epochs = epochs_for_shot_count(n_shot)
                    metrics = train_and_eval(model, train_loader, test_loader, epochs=epochs, lr=lr)

                    row = {"model": model_name, "shots": shots_label, "degradation": degradation,
                           "seed": seed, "lr": lr, **metrics}
                    writer.writerow(row)
                    f.flush()
                    print(f"  -> acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}")

    f.close()
    print(f"\nFinal rerun complete. Results in {FINAL_CSV}")
    print("Next: merge this with small_cnn's original results_raw.csv rows to")
    print("rebuild a clean, consistent Table I (small_cnn stays at lr=1e-3, its")
    print("original appropriate rate; mobilenet/vit use this new lr=1e-4 data).")


if __name__ == "__main__":
    run_final_rerun()
