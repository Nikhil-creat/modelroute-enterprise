"""Inference wrapper for the attachment CNN: label, calibrated confidence (C_vision), features.

Usage:
    python -m vision_cnn.classify_image data/images/UI_ERROR/TKT-000001.png
"""
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

import config
from vision_cnn.train_cnn import build_model, eval_transform


@dataclass
class VisionResult:
    label: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)
    features: list[float] | None = None


class CNNClassifier:
    def __init__(self, weights_path: Path | None = None, meta_path: Path | None = None,
                 device: str | None = None) -> None:
        weights_path = weights_path or config.CNN_WEIGHTS_PATH
        meta_path = meta_path or config.CNN_META_PATH
        if not weights_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"CNN artifacts missing; run `python -m vision_cnn.train_cnn` ({weights_path})")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.classes: list[str] = self.meta["classes"]
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(self.meta["backbone"], len(self.classes), pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.transform = eval_transform(int(self.meta["image_size"]))
        self._lock = threading.Lock()

    @torch.inference_mode()
    def classify(self, image: str | Path | Image.Image, return_features: bool = False) -> VisionResult:
        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                img = handle.convert("RGB")
        else:
            img = image.convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with self._lock:
            feats = self.model.features(x)
            logits = self.model.head(feats)
        probs = torch.softmax(logits[0].float() / self.temperature, dim=0).cpu()
        top = int(probs.argmax())
        return VisionResult(
            label=self.classes[top], confidence=float(probs[top]),
            probs={c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            features=feats[0].float().cpu().tolist() if return_features else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--features", action="store_true", help="also print the penultimate feature vector size")
    args = parser.parse_args()
    result = CNNClassifier().classify(args.image, return_features=args.features)
    print(json.dumps({"label": result.label, "confidence": round(result.confidence, 4), "probs": result.probs,
                      "feature_dim": len(result.features) if result.features else None}, indent=2))


if __name__ == "__main__":
    main()
