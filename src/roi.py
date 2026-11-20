from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np


@dataclass
class ROI:
    image: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float


class LetterboxMeta(NamedTuple):
    """Padding metadata returned by :func:`letterbox_resize`."""
    pad_top: int
    pad_left: int
    scale: float


def letterbox_resize(image: np.ndarray, target_size: int) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize *image* to *target_size* × *target_size* with aspect-ratio-preserving
    letterbox padding (black borders).

    Returns the padded image and a :class:`LetterboxMeta` that lets you undo the
    padding later with :func:`remove_letterbox_padding`.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, image.shape[2]) if image.ndim == 3 else (target_size, target_size), dtype=image.dtype)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized
    return canvas, LetterboxMeta(pad_top=pad_top, pad_left=pad_left, scale=scale)


def remove_letterbox_padding(
    mask: np.ndarray,
    pad_top: int,
    pad_left: int,
    scale: float,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """Undo letterbox padding on a segmentation *mask* and return a mask
    whose spatial dimensions match the original ROI (*orig_h* × *orig_w*).

    Parameters
    ----------
    mask:
        Binary mask at the padded *target_size* × *target_size* resolution.
    pad_top, pad_left:
        Pixel offsets that were applied during letterboxing.
    scale:
        The uniform scale factor used when letterboxing.
    orig_h, orig_w:
        Height and width of the original (un-padded, un-scaled) image.
    """
    scaled_h = int(round(orig_h * scale))
    scaled_w = int(round(orig_w * scale))
    # Crop away the padding to recover the scaled content region
    cropped = mask[pad_top: pad_top + scaled_h, pad_left: pad_left + scaled_w]
    # Resize back to the original ROI dimensions
    return cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def extract_roi(image: np.ndarray, box: tuple[float, float, float, float], confidence: float = 1.0, padding: float = 0.05) -> ROI:
    """Extract a padded bounding-box crop from *image*.

    Parameters
    ----------
    image:
        Full BGR frame.
    box:
        Absolute pixel coordinates ``(x1, y1, x2, y2)``.
    confidence:
        Detection confidence score to attach to the returned :class:`ROI`.
    padding:
        Fractional margin to add around the box (relative to box size).
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_width * padding))
    y1 = max(0, int(y1 - box_height * padding))
    x2 = min(width, int(x2 + box_width * padding))
    y2 = min(height, int(y2 + box_height * padding))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection box does not overlap the image")
    return ROI(image=image[y1:y2, x1:x2].copy(), box=(x1, y1, x2, y2), confidence=confidence)


def paste_mask(mask: np.ndarray, roi: ROI, output_shape: tuple[int, int]) -> np.ndarray:
    """Place a binary *mask* (already at ROI resolution) back into a full-image canvas.

    Parameters
    ----------
    mask:
        Binary mask whose spatial dimensions match the ROI crop
        ``(roi.box[3] - roi.box[1], roi.box[2] - roi.box[0])``.
    roi:
        The :class:`ROI` the mask was predicted for.
    output_shape:
        ``(height, width)`` of the full output canvas.
    """
    output_height, output_width = output_shape
    x1, y1, x2, y2 = roi.box
    # Ensure the mask is at the exact ROI pixel size before pasting
    roi_h, roi_w = y2 - y1, x2 - x1
    if mask.shape[:2] != (roi_h, roi_w):
        mask = cv2.resize(mask.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    full_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask.astype(np.uint8)
    return full_mask