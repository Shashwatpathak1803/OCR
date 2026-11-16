"""
OCR Engine Module using PaddleOCR.
Handles text detection, recognition, angle classification, and confidence scoring
for visiting cards with Windows CPU optimization.
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
from PIL import Image

# Disable PaddleX remote model source check for fast offline initialization
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Suppress verbose paddle logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)

_OCR_INSTANCE = None
_OCR_LOCK = threading.RLock()
_ENGINE_METRICS: Dict[str, Any] = {}


def _hardware_profile() -> Dict[str, Any]:
    import psutil

    memory = psutil.virtual_memory()
    return {
        "cpu_cores": os.cpu_count() or 1,
        "ram_total_gb": round(memory.total / (1024 ** 3), 2),
        "ram_available_gb": round(memory.available / (1024 ** 3), 2),
        "gpu_available": False,
    }


def _cached_model_dir(model_name: str) -> str:
    model_dir = Path.home() / ".paddlex" / "official_models" / model_name
    return str(model_dir) if model_dir.is_dir() else ""


def _apply_paddle_cpu_patch():
    """
    Patch Paddlex static runner config on Windows CPU to prevent oneDNN PIR
    attribute conversion exception.
    """
    try:
        import paddlex.inference.models.runners.paddle_static.runner as psr
        if not getattr(psr, "_patched_run_mode", False):
            orig_resolve = psr.resolve_paddle_static_engine_config

            def safe_resolve(*args, **kwargs):
                cfg = orig_resolve(*args, **kwargs)
                if cfg.get("device_type", "cpu") == "cpu":
                    cfg["run_mode"] = "paddle"
                return cfg

            psr.resolve_paddle_static_engine_config = safe_resolve
            psr._patched_run_mode = True
    except Exception as e:
        logging.warning("Could not apply Paddle CPU patch: %s", e)


def get_ocr_engine(lang: str = "hi"):
    """
    Get or create singleton PaddleOCR instance.
    Uses 'hi' (Devanagari model) which natively recognizes both Hindi and English/Latin scripts.
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        with _OCR_LOCK:
            if _OCR_INSTANCE is None:
                init_started = time.perf_counter()
                _apply_paddle_cpu_patch()
                from paddleocr import PaddleOCR
                import paddle

                profile = _hardware_profile()
                try:
                    profile["gpu_available"] = bool(
                        paddle.device.is_compiled_with_cuda()
                        and paddle.device.cuda.device_count() > 0
                    )
                except Exception as exc:
                    logging.warning("GPU detection failed; using CPU: %s", exc)

                device = "gpu:0" if profile["gpu_available"] else "cpu"
                mobile_detector = _cached_model_dir("PP-OCRv4_mobile_det")
                if profile["gpu_available"]:
                    detector_name = "PP-OCRv5_server_det"
                else:
                    detector_name = "PP-OCRv4_mobile_det" if mobile_detector else ""

                engine_options: Dict[str, Any] = {
                    "lang": lang,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "device": device,
                }
                if detector_name:
                    engine_options["text_detection_model_name"] = detector_name
                    engine_options["text_recognition_model_name"] = (
                        "devanagari_PP-OCRv5_mobile_rec"
                    )

                try:
                    _OCR_INSTANCE = PaddleOCR(**engine_options)
                except Exception as exc:
                    if device == "cpu":
                        raise RuntimeError(
                            f"Failed to initialize PaddleOCR on CPU: {exc}"
                        ) from exc
                    logging.warning(
                        "GPU OCR initialization failed; retrying with CPU: %s", exc
                    )
                    engine_options["device"] = "cpu"
                    if mobile_detector:
                        engine_options["text_detection_model_name"] = (
                            "PP-OCRv4_mobile_det"
                        )
                        detector_name = "PP-OCRv4_mobile_det"
                    else:
                        engine_options.pop("text_detection_model_name", None)
                        engine_options.pop("text_recognition_model_name", None)
                        detector_name = "PaddleOCR language default"
                    _OCR_INSTANCE = PaddleOCR(**engine_options)
                    device = "cpu"

                _ENGINE_METRICS.update(profile)
                _ENGINE_METRICS.update({
                    "device": device,
                    "detector_model": detector_name or "PaddleOCR language default",
                    "recognizer_model": "devanagari_PP-OCRv5_mobile_rec",
                    "initialization_s": round(
                        time.perf_counter() - init_started, 4
                    ),
                })
    return _OCR_INSTANCE


def get_engine_metrics() -> Dict[str, Any]:
    """Return a copy of the active engine's initialization and hardware metrics."""
    return dict(_ENGINE_METRICS)


def _predict_with_stage_timings(ocr_engine: Any, image: np.ndarray):
    pipeline = getattr(ocr_engine, "paddlex_pipeline", None)
    pipeline = getattr(pipeline, "_pipeline", None)
    if pipeline is None:
        started = time.perf_counter()
        results = ocr_engine.predict(image)
        return results, {
            "detection_s": None,
            "recognition_s": None,
            "ocr_s": round(time.perf_counter() - started, 4),
        }

    timed_models = (
        ("text_det_model", "detection_s"),
        ("text_rec_model", "recognition_s"),
    )
    stage_times: Dict[str, float] = {}
    originals = []
    with _OCR_LOCK:
        try:
            for model_attr, timing_key in timed_models:
                model = getattr(pipeline, model_attr, None)
                original_process = getattr(model, "process", None)
                if model is None or original_process is None:
                    continue
                originals.append((model, original_process))
                stage_times[timing_key] = 0.0

                def timed_process(*args, _process=original_process,
                                  _key=timing_key, **kwargs):
                    started = time.perf_counter()
                    result = _process(*args, **kwargs)
                    stage_times[_key] += time.perf_counter() - started
                    return result

                model.process = timed_process

            started = time.perf_counter()
            results = ocr_engine.predict(image)
            ocr_time = time.perf_counter() - started
        finally:
            for model, original_process in originals:
                model.process = original_process

    return results, {
        "detection_s": round(stage_times["detection_s"], 4)
        if "detection_s" in stage_times else None,
        "recognition_s": round(stage_times["recognition_s"], 4)
        if "recognition_s" in stage_times else None,
        "ocr_s": round(ocr_time, 4),
    }


def extract_text_from_image(
    image_input: Union[np.ndarray, Image.Image, str],
    ocr_engine=None
) -> Dict[str, Any]:
    """
    Perform OCR on a visiting card image.

    Returns:
    {
        "success": bool,
        "raw_text": str,  # Full text newline-separated
        "lines": [
            {
                "text": str,
                "confidence": float,
                "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            },
            ...
        ],
        "average_confidence": float,
        "line_count": int,
        "error_message": Optional[str]
    }
    """
    if ocr_engine is None:
        ocr_engine = get_ocr_engine()

    # Convert PIL Image to BGR numpy array if needed
    if isinstance(image_input, Image.Image):
        cv_img = np.array(image_input.convert("RGB"))
        # RGB to BGR
        cv_img = cv_img[:, :, ::-1]
    elif isinstance(image_input, np.ndarray):
        cv_img = image_input
    elif isinstance(image_input, str):
        import cv2
        cv_img = cv2.imread(image_input)
        if cv_img is None:
            return {
                "success": False,
                "raw_text": "",
                "lines": [],
                "average_confidence": 0.0,
                "line_count": 0,
                "error_message": f"Failed to read image from path: {image_input}"
            }
    else:
        return {
            "success": False,
            "raw_text": "",
            "lines": [],
            "average_confidence": 0.0,
            "line_count": 0,
            "error_message": f"Unsupported input type for OCR: {type(image_input)}"
        }

    try:
        # Run inference
        # In PaddleOCR 3.x, predict(img) or ocr(img) is available
        with _OCR_LOCK:
            results, timing = _predict_with_stage_timings(ocr_engine, cv_img)
        lines: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            first_res = results[0]

            # Case A: PaddleX OCRResult object / dict (PaddleOCR 3.x)
            if hasattr(first_res, "get") or isinstance(first_res, dict):
                rec_texts = first_res.get("rec_texts", [])
                rec_scores = first_res.get("rec_scores", [])
                rec_polys = first_res.get("rec_polys", [])
                rec_boxes = first_res.get("rec_boxes", [])

                for idx, text in enumerate(rec_texts):
                    clean_text = str(text).strip()
                    if not clean_text:
                        continue
                    score = float(rec_scores[idx]) if idx < len(rec_scores) else 1.0
                    box_coords = []
                    if idx < len(rec_polys):
                        poly = rec_polys[idx]
                        box_coords = poly.tolist() if hasattr(poly, "tolist") else list(poly)
                    elif idx < len(rec_boxes):
                        box = rec_boxes[idx]
                        box_coords = box.tolist() if hasattr(box, "tolist") else list(box)

                    lines.append({
                        "text": clean_text,
                        "confidence": round(score, 4),
                        "box": box_coords
                    })

            # Case B: Traditional PaddleOCR list of [[box, (text, score)], ...]
            elif isinstance(first_res, list):
                for item in first_res:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        box, text_info = item
                        if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                            text, score = text_info
                            clean_text = str(text).strip()
                            if clean_text:
                                lines.append({
                                    "text": clean_text,
                                    "confidence": round(float(score), 4),
                                    "box": box if isinstance(box, list) else (box.tolist() if hasattr(box, "tolist") else [])
                                })

        raw_text = "\n".join(line["text"] for line in lines)
        avg_conf = (
            round(sum(line["confidence"] for line in lines) / len(lines), 4)
            if lines else 0.0
        )

        return {
            "success": True,
            "raw_text": raw_text,
            "lines": lines,
            "average_confidence": avg_conf,
            "line_count": len(lines),
            "error_message": None if lines else "No text recognized on the visiting card.",
            "timings": timing,
        }

    except Exception as exc:
        logging.exception("OCR recognition error: %s", exc)
        return {
            "success": False,
            "raw_text": "",
            "lines": [],
            "average_confidence": 0.0,
            "line_count": 0,
            "error_message": f"PaddleOCR processing error: {str(exc)}",
            "timings": {},
        }
