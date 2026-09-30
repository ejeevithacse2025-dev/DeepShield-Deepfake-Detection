import io
import cv2
import numpy as np
from PIL import Image
from typing import Union, BinaryIO
from src.config import INPUT_SIZE

def load_image_rgb(source: Union[str, bytes, BinaryIO, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Loads any image source and returns an RGB numpy array (uint8).
    Supports file path, bytes, file-like buffer, PIL Image, or numpy array.
    """
    if isinstance(source, np.ndarray):
        if len(source.shape) == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        elif source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        elif source.shape[2] == 3:
            return source
        raise ValueError(f"Unsupported image array shape: {source.shape}")

    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"), dtype=np.uint8)

    if isinstance(source, (bytes, bytearray)):
        image_bytes = np.frombuffer(source, np.uint8)
        img_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from provided bytes.")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode()
        return load_image_rgb(data)

    if isinstance(source, str):
        img_bgr = cv2.imread(source)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not open image file at: {source}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    raise TypeError(f"Unsupported image source type: {type(source)}")

def prepare_face_for_model(face_rgb: np.ndarray, target_size: tuple = INPUT_SIZE) -> np.ndarray:
    """
    Resizes the cropped face to target_size (224, 224) and formats as
    a batch tensor of shape (1, 224, 224, 3) in float32.
    
    Note: The DeepShield MobileNetV2 model contains built-in normalization layers
    (TrueDivide by 127.5 and Subtract 1.0), so the input pixels must be in range [0, 255].
    """
    if face_rgb.shape[:2] != target_size:
        # If downsampling, INTER_AREA produces cleaner results; if upsampling, INTER_LINEAR
        interpolation = cv2.INTER_AREA if (face_rgb.shape[0] > target_size[0]) else cv2.INTER_LINEAR
        resized = cv2.resize(face_rgb, (target_size[1], target_size[0]), interpolation=interpolation)
    else:
        resized = face_rgb

    tensor = np.expand_dims(resized.astype(np.float32), axis=0)
    return tensor
