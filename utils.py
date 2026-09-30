import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

def create_synthetic_face_image(is_fake: bool = False, size: tuple = (400, 400)) -> np.ndarray:
    """
    Creates a synthetic test image with a recognizable face-like structure
    (oval face, eyes, nose, mouth) using OpenCV primitives.
    Useful for local testing and validation without downloading external assets.
    """
    w, h = size
    # Background
    img = np.full((h, w, 3), (230, 235, 240), dtype=np.uint8)

    # Face oval
    center = (w // 2, h // 2)
    axes = (int(w * 0.28), int(h * 0.38))
    skin_color = (210, 180, 150)
    cv2.ellipse(img, center, axes, 0, 0, 360, skin_color, -1)

    # Eyes
    eye_y = int(h * 0.42)
    left_eye_x = int(w * 0.40)
    right_eye_x = int(w * 0.60)
    eye_radius = int(w * 0.04)
    
    cv2.circle(img, (left_eye_x, eye_y), eye_radius, (255, 255, 255), -1)
    cv2.circle(img, (right_eye_x, eye_y), eye_radius, (255, 255, 255), -1)
    cv2.circle(img, (left_eye_x, eye_y), eye_radius // 2, (50, 40, 30), -1)
    cv2.circle(img, (right_eye_x, eye_y), eye_radius // 2, (50, 40, 30), -1)

    # Eyebrows
    cv2.line(img, (left_eye_x - eye_radius, eye_y - 15), (left_eye_x + eye_radius, eye_y - 15), (60, 40, 20), 3)
    cv2.line(img, (right_eye_x - eye_radius, eye_y - 15), (right_eye_x + eye_radius, eye_y - 15), (60, 40, 20), 3)

    # Nose
    nose_pts = np.array([
        [w // 2, int(h * 0.48)],
        [int(w * 0.47), int(h * 0.58)],
        [int(w * 0.53), int(h * 0.58)]
    ], np.int32)
    cv2.polylines(img, [nose_pts], isClosed=False, color=(170, 140, 110), thickness=2)

    # Mouth
    mouth_center = (w // 2, int(h * 0.68))
    mouth_axes = (int(w * 0.12), int(h * 0.05))
    cv2.ellipse(img, mouth_center, mouth_axes, 0, 0, 180, (180, 70, 70), -1)

    if is_fake:
        # Add slight boundary warping / blending artifacts simulating deepfake artifacts
        warp_overlay = np.zeros_like(img)
        cv2.rectangle(warp_overlay, (int(w * 0.35), int(h * 0.35)), (int(w * 0.65), int(h * 0.75)), (50, 150, 255), -1)
        img = cv2.addWeighted(img, 0.92, warp_overlay, 0.08, 0)
        # Noise
        noise = np.random.normal(0, 12, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

def create_synthetic_video(output_path: str, num_frames: int = 30, fps: int = 15, is_fake: bool = False):
    """
    Creates a sample MP4 video for testing video analysis functionality.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (400, 400))
    for i in range(num_frames):
        # Slightly alter the face position across frames to simulate motion
        frame = create_synthetic_face_image(is_fake=is_fake, size=(400, 400))
        # Add frame counter text
        cv2.putText(frame, f"Frame {i+1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
