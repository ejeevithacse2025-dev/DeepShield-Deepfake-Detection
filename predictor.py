import os
# Ensure Keras uses PyTorch backend
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from typing import Union, List, Dict, Optional, Any
from pathlib import Path

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    LABEL_FAKE,
    LABEL_REAL
)
from src.face_detector import FaceDetector
from src.preprocessor import load_image_rgb, prepare_face_for_model

class DeepShieldPredictor:
    """Core prediction and inference engine for DeepShield deepfake detection."""

    _instance = None

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, threshold: float = DEFAULT_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.face_detector = FaceDetector()
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DeepShield model file not found at: {model_path}. "
                f"Please verify deepshield_model_final.keras is in models/ or project root."
            )
        # Load Keras 3 model
        model = keras.models.load_model(model_path)
        return model

    def predict_tensor(self, tensor: np.ndarray) -> float:
        """
        Runs model forward pass on a preprocessed face tensor of shape (1, 224, 224, 3).
        Returns raw sigmoid probability of the face being FAKE in [0.0, 1.0].
        """
        raw_pred = self.model(tensor, training=False)
        
        # Handle PyTorch tensor or numpy output
        if hasattr(raw_pred, "detach"):
            prob = float(raw_pred.detach().cpu().numpy().squeeze())
        elif hasattr(raw_pred, "numpy"):
            prob = float(raw_pred.numpy().squeeze())
        else:
            prob = float(np.array(raw_pred).squeeze())

        # Clip probability to [0.0, 1.0]
        return float(np.clip(prob, 0.0, 1.0))

    def classify_probability(self, fake_prob: float, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Classifies fake probability based on threshold:
        - >= threshold -> FAKE (confidence = fake_prob * 100)
        - < threshold  -> REAL (confidence = (1 - fake_prob) * 100)
        """
        t = threshold if threshold is not None else self.threshold
        is_fake = fake_prob >= t
        label = LABEL_FAKE if is_fake else LABEL_REAL
        confidence = (fake_prob if is_fake else (1.0 - fake_prob)) * 100.0

        return {
            "label": label,
            "is_fake": is_fake,
            "confidence": round(confidence, 2),
            "fake_prob": round(fake_prob, 4),
            "real_prob": round(1.0 - fake_prob, 4),
            "threshold": t
        }

    def predict_image(self, image_source, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Full pipeline for a single image:
        1. Loads image to RGB
        2. Detects face(s) or falls back to center crop
        3. Normalizes and scores each face with the DeepShield model
        4. Annotates bounding boxes on original image
        5. Returns structured results dictionary
        """
        t = threshold if threshold is not None else self.threshold
        image_rgb = load_image_rgb(image_source)
        faces_detected, is_fallback = self.face_detector.get_face_or_fallback(image_rgb)

        face_results = []
        for face_info in faces_detected:
            cropped = face_info["face_rgb"]
            tensor = prepare_face_for_model(cropped)
            fake_prob = self.predict_tensor(tensor)
            classification = self.classify_probability(fake_prob, threshold=t)

            face_res = {
                "box": face_info["box"],
                "raw_box": face_info["raw_box"],
                "face_rgb": cropped,
                "is_fallback": face_info["is_fallback"],
                **classification
            }
            face_results.append(face_res)

        # Aggregate overall verdict:
        # If any face is flagged as FAKE, overall verdict is FAKE; otherwise REAL
        max_fake_prob = max(f["fake_prob"] for f in face_results)
        overall_classification = self.classify_probability(max_fake_prob, threshold=t)

        # Annotate image with bounding boxes
        annotated_image = self.face_detector.annotate_image(image_rgb, face_results)

        return {
            "overall_label": overall_classification["label"],
            "overall_is_fake": overall_classification["is_fake"],
            "overall_confidence": overall_classification["confidence"],
            "overall_fake_prob": overall_classification["fake_prob"],
            "overall_real_prob": overall_classification["real_prob"],
            "num_faces": len(face_results) if not is_fallback else 0,
            "is_fallback_used": is_fallback,
            "faces": face_results,
            "annotated_image": annotated_image,
            "original_image": image_rgb
        }
