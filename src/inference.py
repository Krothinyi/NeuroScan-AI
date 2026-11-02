
# ==========================================
# NEUROSCAN AI - STAGE 3 INFERENCE PIPELINE
# ==========================================

import json
from pathlib import Path

import torch
from PIL import Image

from src.dataset import CLASSES
from src.model import build_resnet18
from src.preprocessing import eval_transform
from src.explainability import GradCAM


class NeuroScanPredictor:

    def __init__(self, project_dir, device="cpu"):

        import math

        self.project_dir = Path(project_dir)
        self.device = torch.device(device)

        checkpoint_path = (
            self.project_dir
            / "checkpoints/resnet18_v2/resnet18_v2_best.pth"
        )

        calibration_path = (
            self.project_dir
            / "checkpoints/resnet18_v2/calibration_v2.json"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint.get("architecture") != "resnet18":
            raise ValueError("Unexpected architecture")

        if list(checkpoint.get("classes", [])) != list(CLASSES):
            raise ValueError("Class mapping mismatch")

        self.model = build_resnet18(pretrained=False)
        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_stage = "V2"
        self.model_trial = 1
        self.model_epoch = int(checkpoint["epoch"])

        with open(calibration_path, "r", encoding="utf-8") as f:
            calibration = json.load(f)

        calibrated_path = Path(calibration["checkpoint"])
        if not calibrated_path.is_absolute():
            calibrated_path = self.project_dir / calibrated_path

        if calibrated_path.resolve() != checkpoint_path.resolve():
            raise ValueError("Calibration checkpoint mismatch")

        if int(calibration["checkpoint_epoch"]) != self.model_epoch:
            raise ValueError("Calibration epoch mismatch")

        self.temperature = float(calibration["temperature"])

        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid temperature")

        self.review_threshold = 0.97

    # ======================================
    # PREDICTION
    # ======================================

    def predict(self, image_path, explain=True):

        with Image.open(image_path) as img:
            image = img.convert("RGB").copy()

        # Deterministic preprocessing
        image_tensor = eval_transform(image)

        batch = image_tensor.unsqueeze(0).to(
            self.device
        )

        # ==================================
        # CLASSIFICATION
        # ==================================

        with torch.inference_mode():

            logits = self.model(batch)

            calibrated_probs = torch.softmax(
                logits / self.temperature,
                dim=1
            )[0]

        predicted_idx = int(
            calibrated_probs.argmax().item()
        )

        probabilities = {
            class_name: float(
                calibrated_probs[i].item()
            )
            for i, class_name in enumerate(CLASSES)
        }

        
        # Get predicted class
        predicted_class = CLASSES[predicted_idx]

        # Maximum calibrated probability
        max_probability = probabilities[predicted_class]

        # ==================================
        # CONFIDENCE-BASED REVIEW FLAG
        # ==================================

        review_recommended = (
            max_probability < self.review_threshold
        )

        review_reason = (
            "Model confidence is below the review threshold."
            if review_recommended
            else "No low-confidence review flag triggered."
        )

        # ==================================
        # FINAL PREDICTION RESULT
        # ==================================

        result = {
            "model_stage": self.model_stage,
            "model_epoch": self.model_epoch,
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "max_probability": max_probability,
            "temperature": self.temperature,
            "review_threshold": self.review_threshold,
            "review_recommended": review_recommended,
            "review_reason": review_reason
        }




        # ==================================
        # GRAD-CAM EXPLANATION
        # ==================================

        if explain:

            gradcam = GradCAM(
                model=self.model,
                target_layer=self.model.layer4[-1]
            )

            try:
                heatmap, _ = gradcam.generate(
                    batch,
                    class_idx=predicted_idx
                )

            finally:
                gradcam.close()

            result["heatmap"] = heatmap

        return result
