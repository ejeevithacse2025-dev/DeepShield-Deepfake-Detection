import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.config import FACE_PADDING_RATIO, MIN_FACE_SIZE

class FaceDetector:
    """Detects and extracts faces from images using OpenCV Haar Cascades with padding."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from {cascade_path}")

    def detect_faces(self, image_rgb: np.ndarray, padding_ratio: float = FACE_PADDING_RATIO) -> List[Dict]:
        """
        Detect all frontal faces in an RGB image.
        Returns list of dicts with 'box', 'raw_box', and 'face_rgb'.
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        detected = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces = []
        for (x, y, fw, fh) in detected:
            # Apply padding margin around the face
            pad_w = int(fw * padding_ratio)
            pad_h = int(fh * padding_ratio)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w)
            y2 = min(h, y + fh + pad_h)

            face_crop = image_rgb[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue

            faces.append({
                "box": (x1, y1, x2 - x1, y2 - y1),
                "raw_box": (x, y, fw, fh),
                "face_rgb": face_crop,
                "is_fallback": False
            })

        return faces

    def get_face_or_fallback(self, image_rgb: np.ndarray) -> Tuple[List[Dict], bool]:
        """
        Detect faces in image. If none found, return a centered square crop as fallback.
        Returns (faces_list, is_fallback_used).
        """
        faces = self.detect_faces(image_rgb)
        if faces:
            return faces, False

        # Fallback: square center crop
        h, w = image_rgb.shape[:2]
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        
        fallback_crop = image_rgb[start_y:start_y + crop_size, start_x:start_x + crop_size]
        fallback_face = {
            "box": (start_x, start_y, crop_size, crop_size),
            "raw_box": (start_x, start_y, crop_size, crop_size),
            "face_rgb": fallback_crop,
            "is_fallback": True
        }
        return [fallback_face], True

    @staticmethod
    def annotate_image(image_rgb: np.ndarray, predictions: List[Dict]) -> np.ndarray:
        """
        Draws bounding boxes and classification labels on the image.
        predictions is a list of dicts with 'box', 'label', 'confidence', 'fake_prob'.
        """
        annotated = image_rgb.copy()
        for item in predictions:
            box = item.get("box")
            if not box:
                continue
            x, y, w, h = box
            label = item.get("label", "UNKNOWN")
            confidence = item.get("confidence", 0.0)

            # Red for FAKE, Green for REAL, Amber for fallback/warning
            if label == "FAKE":
                color = (239, 68, 68)   # Red RGB
            else:
                color = (34, 197, 94)   # Green RGB

            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

            # Label text
            text = f"{label} ({confidence:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw background box for text for readability
            text_y = max(y - 8, text_h + 10)
            cv2.rectangle(
                annotated,
                (x, text_y - text_h - 4),
                (x + text_w + 8, text_y + 4),
                color,
                -1
            )
            cv2.putText(
                annotated,
                text,
                (x + 4, text_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated
