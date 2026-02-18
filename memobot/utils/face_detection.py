"""Shared face detection for gatekeeping face_database: only save images that contain a detectable face."""
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# Minimum width/height for a detection to count as a valid face (avoids tiny false positives).
MIN_FACE_SIZE = 60


def _get_cascade():
    """Load OpenCV Haar cascade for frontal face. Returns None if cv2 unavailable or load fails."""
    if not CV2_AVAILABLE:
        return None
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None
    return cascade


def detect_faces_in_image(
    image_path: Path,
    min_face_size: int = MIN_FACE_SIZE,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image file using OpenCV Haar cascade.
    Returns list of (x, y, w, h) for each face with w,h >= min_face_size.
    Returns [] if no face, file missing, or cv2 unavailable.
    """
    if not CV2_AVAILABLE:
        return []
    path = Path(image_path)
    if not path.exists():
        return []
    img = cv2.imread(str(path))
    if img is None:
        return []
    return detect_faces_in_frame(img, min_face_size=min_face_size)


def detect_faces_in_frame(
    frame,
    min_face_size: int = MIN_FACE_SIZE,
):
    """
    Detect faces in a BGR frame (numpy array). Returns list of (x, y, w, h).
    Only returns faces with width and height >= min_face_size.
    """
    if not CV2_AVAILABLE or frame is None or frame.size == 0:
        return []
    cascade = _get_cascade()
    if cascade is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    result = []
    for (x, y, w, h) in faces:
        if w >= min_face_size and h >= min_face_size:
            result.append((int(x), int(y), int(w), int(h)))
    return result


def image_contains_face(
    image_path: Path,
    min_face_size: int = MIN_FACE_SIZE,
) -> bool:
    """Return True if the image has at least one detectable face of sufficient size."""
    return len(detect_faces_in_image(image_path, min_face_size=min_face_size)) > 0
