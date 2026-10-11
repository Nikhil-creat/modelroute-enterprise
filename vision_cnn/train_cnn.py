"""Train the ticket-attachment CNN (ResNet18 / MobileNetV3 / small custom CNN) in PyTorch.

Reads data/splits/{train,val,test}.jsonl, trains on tickets that have an attachment, keeps the
best epoch by validation macro-F1, fits a temperature on validation logits (so the confidence
C_vision is calibrated) and writes artifacts/cnn/cnn_best.pt + cnn_meta.json.

Usage:
    python -m vision_cnn.train_cnn --backbone resnet18 --epochs 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import config

CLASSES = config.VISUAL_CLASSES


class VisionNet(nn.Module):
    """Feature extractor (``backbone``) + linear classifier (``head``)."""

    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


def build_model(backbone: str, num_classes: int, pretrained: bool = False) -> VisionNet:
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3,
                             m.layer4, m.avgpool, nn.Flatten())
        return VisionNet(body, 512, num_classes)
    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=weights)
        return VisionNet(nn.Sequential(m.features, m.avgpool, nn.Flatten()), 576, num_classes)
    if backbone == "custom":
        body = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 128),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten())
        return VisionNet(body, 128, num_classes)
    raise ValueError(f"unknown backbone: {backbone}")


def eval_transform(size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)])


def train_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=4, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)])


class AttachmentDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: transforms.Compose) -> None:
        self.rows = [r for r in rows if r.get("visual_label") and r.get("image_path")]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(config.resolve_data_path(row["image_path"])) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, CLASSES.index(row["visual_label"])


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> dict[str, Any]:
    probs = torch.softmax(logits / temperature, dim=1)
    pred = probs.argmax(1).numpy()
    y = labels.numpy()
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(len(CLASSES))), zero_division=0)),
        "mean_confidence": float(probs.max(1).values.mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(CLASSES)))).tolist(),
    }


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(min(10.0, max(0.05, log_t.exp().item())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default=config.CNN_BACKBONE, choices=["resnet18", "mobilenet_v3_small", "custom"])
    parser.add_argument("--epochs", type=int, default=config.CNN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CNN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.CNN_LR)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = min(4, os.cpu_count() or 1)

    def loader(path, tf, shuffle):
        ds = AttachmentDataset(config.read_jsonl(path), tf)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=workers,
                          pin_memory=device.type == "cuda")

    train_dl = loader(config.TRAIN_PATH, train_transform(args.image_size), True)
    val_dl = loader(config.VAL_PATH, eval_transform(args.image_size), False)
    test_dl = loader(config.TEST_PATH, eval_transform(args.image_size), False)
    print(f"images  train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}  device={device}")

    try:
        model = build_model(args.backbone, len(CLASSES), pretrained=args.pretrained)
    except Exception as exc:  # offline machines cannot download ImageNet weights
        print(f"pretrained weights unavailable ({exc}); training from scratch")
        model = build_model(args.backbone, len(CLASSES), pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        val_logits, val_labels = predict_logits(model, val_dl, device)
        val = metrics_from_logits(val_logits, val_labels)
        print(f"epoch {epoch:02d} loss={running / len(train_dl.dataset):.4f} "
              f"val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} ({time.time() - start:.0f}s)")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_logits, val_labels = predict_logits(model, val_dl, device)
    temperature = fit_temperature(val_logits, val_labels)
    test_logits, test_labels = predict_logits(model, test_dl, device)
    meta = {
        "classes": CLASSES, "backbone": args.backbone, "image_size": args.image_size,
        "temperature": temperature,
        "val": metrics_from_logits(val_logits, val_labels, temperature),
        "test": metrics_from_logits(test_logits, test_labels, temperature),
    }
    torch.save(best_state, config.CNN_WEIGHTS_PATH)
    config.CNN_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {config.CNN_WEIGHTS_PATH}  T={temperature:.3f}")
    print(f"test accuracy={meta['test']['accuracy']:.4f} macro_f1={meta['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
