"""
Image Preprocessing Module for Visiting Card OCR.
Provides OpenCV-based image enhancement, perspective correction,
rotation deskewing, and contrast optimization.
"""

import io
import math
import time
from typing import Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image


def load_image(image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]) -> Tuple[Image.Image, np.ndarray]:
    """
    Load image from various input types and return both PIL Image (RGB) and OpenCV image (BGR).
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img

    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:  # Grayscale
            cv_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            pil_img = Image.fromarray(image_input)
        elif image_input.shape[2] == 4:  # RGBA / BGRA
            cv_img = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        else:
            cv_img = image_input.copy()
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        return pil_img, cv_img

    if isinstance(image_input, (bytes, io.BytesIO)):
        if isinstance(image_input, io.BytesIO):
            image_bytes = image_input.getvalue()
        else:
            image_bytes = image_input
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img

    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def resize_proportional(img: np.ndarray, max_dim: int = 1800, min_dim: int = 800) -> Tuple[np.ndarray, float]:
    """
    Resize image proportionally:
    - Downscale if largest dimension exceeds max_dim to speed up OCR and reduce noise.
    - Upscale if largest dimension is below min_dim to preserve text details.
    """
    h, w = img.shape[:2]
    max_side = max(h, w)
    scale = 1.0

    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    elif max_side < min_dim and max_side > 0:
        scale = min_dim / float(max_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized, scale

    return img, scale


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise_image(gray: np.ndarray) -> np.ndarray:
    """Apply bilateral filtering to reduce noise while maintaining sharp edges."""
    return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


def enhance_contrast_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points of a quadrilateral:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest diff (y - x), bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_and_warp_card(cv_img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Attempt to find visiting card boundary rectangle against a background and warp perspective.
    If no valid 4-corner boundary covering 20%-95% of image area is found,
    return original image intact.
    """
    orig_h, orig_w = cv_img.shape[:2]
    total_area = orig_h * orig_w

    # Downscale copy for contour finding
    small_img, scale = resize_proportional(cv_img, max_dim=900, min_dim=400)
    gray = to_grayscale(small_img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_cnt = None
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            # Area must be between 20% and 95% of the total image
            if 0.20 * (small_img.shape[0] * small_img.shape[1]) < area < 0.95 * (small_img.shape[0] * small_img.shape[1]):
                card_cnt = approx
                break

    if card_cnt is None:
        return cv_img, False

    # Scale back points to original resolution
    pts = card_cnt.reshape(4, 2).astype("float32") / scale
    ordered_pts = order_quad_points(pts)

    (tl, tr, br, bl) = ordered_pts
    # Compute width and height of new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(int(height_a), int(height_b))

    # Safety check on aspect ratio: visiting cards are usually ~1.4 to ~2.0 ratio
    if max_w < 100 or max_h < 100:
        return cv_img, False

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(cv_img, M, (max_w, max_h), flags=cv2.INTER_CUBIC)
    return warped, True


def detect_and_correct_skew(gray: np.ndarray, max_angle: float = 30.0) -> Tuple[np.ndarray, float]:
    """
    Detect text skew angle and rotate image to straighten horizontal text lines.
    Rotates only if detected angle is within +/- max_angle and > 0.5 degrees.
    """
    h, w = gray.shape[:2]
    # Invert binary image: text becomes white, background black
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )

    # Find coordinates of all foreground pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return gray, 0.0

    # Determine minimum area bounding box
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle from minAreaRect convention
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle

    # Check threshold bounds
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return gray, 0.0

    # Rotate around image center
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def preprocess_visiting_card(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    auto_perspective: bool = True,
    auto_deskew: bool = True,
    enhance_contrast: bool = True,
    target_max_dim: int = 1800,
) -> Dict[str, Any]:
    """
    Full visiting card preprocessing pipeline.
    Preserves original image and produces an enhanced version for OCR recognition.

    Returns dict containing:
    - 'original_pil': PIL Image (RGB)
    - 'original_cv': OpenCV image (BGR)
    - 'processed_pil': Enhanced PIL Image (RGB)
    - 'processed_cv': Enhanced OpenCV image for OCR (BGR or Grayscale)
    - 'metadata': Information about steps performed
    - 'timings': Image-load and preprocessing durations in seconds
    """
    load_started = time.perf_counter()
    orig_pil, orig_cv = load_image(image_input)
    image_loading_s = time.perf_counter() - load_started
    preprocessing_started = time.perf_counter()
    working_cv = orig_cv.copy()

    metadata = {
        "perspective_corrected": False,
        "deskew_angle": 0.0,
        "resized": False,
        "scale": 1.0,
        "contrast_enhanced": False,
    }

    # Step 1: Perspective correction if enabled
    if auto_perspective:
        warped_cv, was_warped = detect_and_warp_card(working_cv)
        if was_warped:
            working_cv = warped_cv
            metadata["perspective_corrected"] = True

    # Step 2: Proportional resize
    resized_cv, scale = resize_proportional(working_cv, max_dim=target_max_dim, min_dim=800)
    metadata["resized"] = scale != 1.0
    metadata["scale"] = scale
    working_cv = resized_cv

    # Step 3: Grayscale conversion
    gray = to_grayscale(working_cv)

    # Step 4: Denoising
    denoised = denoise_image(gray)

    # Step 5: Contrast enhancement
    if enhance_contrast:
        enhanced = enhance_contrast_clahe(denoised, clip_limit=2.0)
        metadata["contrast_enhanced"] = True
    else:
        enhanced = denoised

    # Step 6: Deskew / Rotation correction
    if auto_deskew:
        straightened, angle = detect_and_correct_skew(enhanced)
        metadata["deskew_angle"] = round(angle, 2)
        enhanced = straightened

    # Convert back to 3-channel BGR for PaddleOCR input compatibility
    processed_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    processed_pil = Image.fromarray(enhanced)
    preprocessing_s = time.perf_counter() - preprocessing_started

    return {
        "original_pil": orig_pil,
        "original_cv": orig_cv,
        "processed_pil": processed_pil,
        "processed_cv": processed_bgr,
        "metadata": metadata,
        "timings": {
            "image_loading_s": round(image_loading_s, 4),
            "preprocessing_s": round(preprocessing_s, 4),
        },
    }
