import cv2
import os
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from src.config import (
    DEFAULT_THRESHOLD,
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_VIDEO_SAMPLE_FPS,
    LABEL_FAKE,
    LABEL_REAL
)
from src.face_detector import FaceDetector
from src.preprocessor import prepare_face_for_model
from src.predictor import DeepShieldPredictor

class VideoProcessor:
    """Extracts frames, analyzes facial deepfake signatures, and computes temporal metrics."""

    def __init__(self, predictor: DeepShieldPredictor):
        self.predictor = predictor
        self.face_detector = predictor.face_detector

    def analyze_video(
        self,
        video_path: str,
        threshold: Optional[float] = None,
        max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
        sample_fps: float = DEFAULT_VIDEO_SAMPLE_FPS,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a video file by sampling frames and running deepfake detection.
        
        Args:
            video_path: Path to the video file.
            threshold: Decision threshold for fake vs real (default 0.50).
            max_frames: Maximum number of frames to sample and analyze.
            sample_fps: Desired sampling rate in frames per second.
            progress_callback: Optional callback func(progress_float_0_to_1, status_str).
        """
        t = threshold if threshold is not None else self.predictor.threshold
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream from {video_path}")

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_video_frames / video_fps if total_video_frames > 0 else 0.0

        # Determine frame indices to sample
        step = max(1, int(video_fps / sample_fps))
        frame_indices = list(range(0, total_video_frames, step))
        if len(frame_indices) > max_frames:
            # Uniformly subsample if too many frames
            selected_idxs = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in selected_idxs]

        if not frame_indices:
            frame_indices = [0]

        frame_results = []
        current_frame_idx = 0
        target_idx_ptr = 0
        total_to_process = len(frame_indices)

        while cap.isOpened() and target_idx_ptr < total_to_process:
            target_idx = frame_indices[target_idx_ptr]
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = target_idx / video_fps

            # Detect face or fallback
            faces, is_fallback = self.face_detector.get_face_or_fallback(frame_rgb)
            
            # Predict for each face in the frame
            frame_face_probs = []
            frame_face_details = []
            for face in faces:
                tensor = prepare_face_for_model(face["face_rgb"])
                fake_prob = self.predictor.predict_tensor(tensor)
                frame_face_probs.append(fake_prob)
                frame_face_details.append({
                    "fake_prob": fake_prob,
                    "box": face["box"],
                    "is_fallback": face["is_fallback"]
                })

            max_frame_fake_prob = max(frame_face_probs) if frame_face_probs else 0.0
            frame_class = self.predictor.classify_probability(max_frame_fake_prob, threshold=t)

            frame_entry = {
                "frame_index": target_idx,
                "timestamp_sec": round(timestamp, 2),
                "fake_prob": round(max_frame_fake_prob, 4),
                "is_fake": frame_class["is_fake"],
                "label": frame_class["label"],
                "confidence": frame_class["confidence"],
                "is_fallback": is_fallback,
                "num_faces": len(faces) if not is_fallback else 0,
                "thumbnail_rgb": cv2.resize(frame_rgb, (160, 90)),
                "face_crop_rgb": cv2.resize(faces[0]["face_rgb"], (112, 112))
            }
            frame_results.append(frame_entry)

            target_idx_ptr += 1
            if progress_callback:
                progress = float(target_idx_ptr) / float(total_to_process)
                progress_callback(
                    progress,
                    f"Analyzing frame {target_idx_ptr}/{total_to_process} ({timestamp:.1f}s)..."
                )

        cap.release()

        if not frame_results:
            raise RuntimeError("No frames could be extracted from the video.")

        # Temporal Aggregation
        all_probs = [f["fake_prob"] for f in frame_results]
        avg_fake_prob = float(np.mean(all_probs))
        max_fake_prob = float(np.max(all_probs))
        min_fake_prob = float(np.min(all_probs))

        fake_frames_count = sum(1 for f in frame_results if f["is_fake"])
        total_analyzed = len(frame_results)
        fake_ratio = fake_frames_count / total_analyzed

        # Weighted video score combining mean and peak anomaly (70% mean, 30% peak)
        video_fake_prob = (0.7 * avg_fake_prob) + (0.3 * max_fake_prob)
        overall_classification = self.predictor.classify_probability(video_fake_prob, threshold=t)

        # Sort suspicious frames (highest fake_prob first)
        suspicious_frames = sorted(frame_results, key=lambda x: x["fake_prob"], reverse=True)

        return {
            "video_path": video_path,
            "duration_sec": round(duration_sec, 2),
            "video_fps": round(video_fps, 2),
            "total_video_frames": total_video_frames,
            "analyzed_frames_count": total_analyzed,
            "overall_label": overall_classification["label"],
            "overall_is_fake": overall_classification["is_fake"],
            "overall_confidence": overall_classification["confidence"],
            "overall_fake_prob": round(video_fake_prob, 4),
            "avg_fake_prob": round(avg_fake_prob, 4),
            "max_fake_prob": round(max_fake_prob, 4),
            "min_fake_prob": round(min_fake_prob, 4),
            "fake_frames_count": fake_frames_count,
            "fake_ratio_percent": round(fake_ratio * 100.0, 1),
            "timeline": frame_results,
            "top_suspicious_frames": suspicious_frames[:6]
        }
