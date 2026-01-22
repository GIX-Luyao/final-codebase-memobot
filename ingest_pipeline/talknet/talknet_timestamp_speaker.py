#!/usr/bin/env python3
"""
TalkNet timestamp speaker + face screenshots (uses pretrained TalkNet demo pipeline).

How it works:
1) Runs TalkNet-ASD demo inference:
      python demoTalkNet.py --videoName <NAME>
   which produces:
      demo/<NAME>/pyavi/video_out.avi  (annotated: green=active speaker, red=non-active)
2) Seeks to a timestamp in that output video.
3) Detects the green/red rectangles from the annotated frame via color thresholding.
4) Uses those boxes to crop faces from the original frame at the same timestamp.
5) Saves:
   - out/<NAME>_t<ms>/frame.jpg
   - out/<NAME>_t<ms>/faces/face_XX.jpg
   - out/<NAME>_t<ms>/speaker_face_XX.jpg (if any green box found)

Requirements:
- You must be inside the TalkNet-ASD repo root (where demoTalkNet.py exists).
- ffmpeg is NOT required for file input (only OpenCV is).
- pip deps: opencv-python, numpy

Notes:
- This leverages TalkNet's own detection/tracking/ASD, so you DON'T need to implement face detection yourself.
- For "streaming", see the bottom: it can sample a stream by recording a short clip (optional).

TalkNet demo output details:
- Run: python demoTalkNet.py --videoName 001
- Output video: demo/001/pyavi/video_out.avi (green active speaker, red non-active) :contentReference[oaicite:1]{index=1}
"""

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    color: str  # "green" or "red"

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def clamp(self, w: int, h: int) -> "Box":
        return Box(
            x1=max(0, min(self.x1, w - 1)),
            y1=max(0, min(self.y1, h - 1)),
            x2=max(0, min(self.x2, w - 1)),
            y2=max(0, min(self.y2, h - 1)),
            color=self.color,
        )


def run_talknet_demo(repo_root: Path, video_name: str, force: bool = False, confidence_threshold: float = -0.5) -> Path:
    """
    Runs: python demoTalkNet.py --videoName <video_name> --confidenceThreshold <threshold>
    Returns path to annotated output video: demo/<video_name>/pyavi/video_out.avi
    """
    demo_out = repo_root / "demo" / video_name / "pyavi" / "video_out.avi"
    if demo_out.exists() and not force:
        return demo_out

    cmd = ["python3", "demoTalkNet.py", "--videoName", video_name, "--confidenceThreshold", str(confidence_threshold)]
    print(f"[TalkNet] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    if not demo_out.exists():
        raise FileNotFoundError(
            f"Expected TalkNet output not found: {demo_out}\n"
            f"TalkNet demo should create demo/{video_name}/pyavi/video_out.avi"
        )
    return demo_out


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def get_video_fps_and_framecount(cap: cv2.VideoCapture) -> Tuple[float, int]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0  # fallback
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return float(fps), n


def read_frame_at_timestamp(cap: cv2.VideoCapture, ts_sec: float) -> Tuple[np.ndarray, int]:
    fps, n = get_video_fps_and_framecount(cap)
    frame_idx = int(round(ts_sec * fps))
    if n > 0:
        frame_idx = max(0, min(frame_idx, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at t={ts_sec:.3f}s (frame {frame_idx})")
    return frame, frame_idx


def _mask_hsv(frame_bgr: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_np = np.array(lower, dtype=np.uint8)
    upper_np = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_np, upper_np)
    # clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask


def find_colored_boxes(annot_bgr: np.ndarray) -> List[Box]:
    """
    Detects TalkNet's green/red rectangles from the annotated frame.
    TalkNet draws boxes as thick outlines (thickness 10) in pure colors:
    - Green: (0, 255, 0) in BGR = HSV ~(60, 255, 255) for active speakers
    - Red: (0, 0, 255) in BGR = HSV ~(0, 255, 255) or (180, 255, 255) for non-active
    
    We use strict thresholds to avoid detecting green/red objects in the scene.
    """
    h, w = annot_bgr.shape[:2]

    # Green (active speaker box): Pure green is HSV ~(60, 255, 255)
    # Use tight range around pure green with high saturation/brightness requirements
    # This avoids detecting green plants, text, etc. in the scene
    green_mask = _mask_hsv(annot_bgr, lower=(50, 200, 200), upper=(70, 255, 255))

    # Red: Pure red is HSV ~(0, 255, 255) or (180, 255, 255)
    # Use tight range with high saturation/brightness to avoid scene objects
    red_mask1 = _mask_hsv(annot_bgr, lower=(0, 200, 200), upper=(10, 255, 255))
    red_mask2 = _mask_hsv(annot_bgr, lower=(170, 200, 200), upper=(180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    boxes: List[Box] = []
    green_boxes = _boxes_from_mask(green_mask, color="green", w=w, h=h)
    red_boxes = _boxes_from_mask(red_mask, color="red", w=w, h=h)
    boxes.extend(green_boxes)
    boxes.extend(red_boxes)

    print(f"[Detection] Found {len(green_boxes)} green boxes, {len(red_boxes)} red boxes (before NMS)")

    # de-duplicate (sometimes edges create multiple contours)
    boxes_before_nms = len(boxes)
    boxes = _nms_boxes(boxes, iou_thresh=0.3)
    print(f"[Detection] After NMS: {len(boxes)} boxes (filtered {boxes_before_nms - len(boxes)})")
    return boxes


def _is_valid_face_box(box: Box, frame_h: int, frame_w: int) -> bool:
    """
    Validate that a box is likely a face, not a subtitle, hand, or other artifact.
    """
    # Check aspect ratio - faces are roughly square to slightly tall
    # Typical face aspect ratio is between 0.7 (tall) and 1.5 (wide)
    box_w = box.x2 - box.x1
    box_h = box.y2 - box.y1
    if box_w == 0 or box_h == 0:
        return False
    aspect_ratio = box_w / box_h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
    
    # Filter out boxes in the bottom 15% of frame (where subtitles usually are)
    bottom_threshold = frame_h * 0.85
    if box.y1 > bottom_threshold:
        return False
    
    # Filter out boxes that are too small (already checked, but double-check)
    min_area = 1200  # Increased from 800
    if box.area < min_area:
        return False
    
    # Filter out boxes that are too large (likely false positives)
    max_area = frame_w * frame_h * 0.3  # No more than 30% of frame
    if box.area > max_area:
        return False
    
    return True


def _boxes_from_mask(mask: np.ndarray, color: str, w: int, h: int) -> List[Box]:
    """
    Extract bounding boxes from color mask.
    TalkNet draws boxes as thick rectangular outlines, so we look for rectangular contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Box] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        # Filter small noise and thin lines:
        if bw * bh < 1200:  # Increased from 800
            continue
        if bw < 30 or bh < 30:  # Increased from 20
            continue
        
        # Basic size validation - ensure it's not too small or degenerate
        if bw == 0 or bh == 0:
            continue
        
        # Expand slightly to capture full face (the box outline is around the face)
        pad = 6
        b = Box(x - pad, y - pad, x + bw + pad, y + bh + pad, color=color).clamp(w, h)
        # Validate it's a reasonable face box
        if _is_valid_face_box(b, h, w):
            out.append(b)
    return out


def _iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / max(1, union)


def _nms_boxes(boxes: List[Box], iou_thresh: float = 0.3) -> List[Box]:
    if not boxes:
        return []
    boxes_sorted = sorted(boxes, key=lambda b: b.area, reverse=True)
    keep: List[Box] = []
    for b in boxes_sorted:
        if all(_iou(b, k) < iou_thresh for k in keep):
            keep.append(b)
    return keep


def draw_boxes(frame: np.ndarray, boxes: List[Box], speaker_box: Optional[Box]) -> np.ndarray:
    out = frame.copy()
    for i, b in enumerate(boxes):
        color = (0, 255, 0) if b.color == "green" else (0, 0, 255)
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        cv2.putText(
            out,
            f"{b.color.upper()}_{i:02d}",
            (b.x1, max(0, b.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if speaker_box is not None:
        cv2.putText(
            out,
            "SPEAKER = GREEN BOX",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out,
            "SPEAKER NOT FOUND (no green box)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def save_outputs(
    out_dir: Path,
    frame_bgr: np.ndarray,
    boxes: List[Box],
    speaker_box: Optional[Box],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faces_dir = out_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    # Save full frame
    frame_path = out_dir / "frame.jpg"
    cv2.imwrite(str(frame_path), frame_bgr)
    print(f"[OK] Saved frame: {frame_path}")

    # Save face crops
    h, w = frame_bgr.shape[:2]
    for i, b in enumerate(boxes):
        bb = b.clamp(w, h)
        # Ensure crop dimensions are valid
        if bb.x2 <= bb.x1 or bb.y2 <= bb.y1:
            print(f"[SKIP] Invalid box dimensions for face_{i:02d}_{b.color}: {bb.x1},{bb.y1} -> {bb.x2},{bb.y2}")
            continue
        crop = frame_bgr[bb.y1:bb.y2, bb.x1:bb.x2]
        # Skip empty or too-small crops
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            print(f"[SKIP] Crop too small or empty for face_{i:02d}_{b.color}: {crop.shape}")
            continue
        crop_path = faces_dir / f"face_{i:02d}_{b.color}.jpg"
        cv2.imwrite(str(crop_path), crop)
        print(f"[OK] Saved face crop: {crop_path} (size: {crop.shape[1]}x{crop.shape[0]})")

    # Save speaker crop (largest green box)
    if speaker_box is not None:
        sb = speaker_box.clamp(w, h)
        scrop = frame_bgr[sb.y1:sb.y2, sb.x1:sb.x2]
        spath = out_dir / "speaker_face.jpg"
        cv2.imwrite(str(spath), scrop)
        print(f"[OK] Saved speaker crop: {spath}")


def ensure_video_in_demo(repo_root: Path, input_video: Path, video_name: str) -> Path:
    """
    TalkNet demo expects the raw video inside ./demo as <video_name>.mp4 or .avi (per README). :contentReference[oaicite:2]{index=2}
    We'll copy/overwrite to demo/<video_name>.<ext>.
    Returns the copied destination path.
    """
    demo_dir = repo_root / "demo"
    demo_dir.mkdir(exist_ok=True)
    ext = input_video.suffix.lower()
    if ext not in [".mp4", ".avi"]:
        raise ValueError("TalkNet demo expects .mp4 or .avi input (per README).")
    dst = demo_dir / f"{video_name}{ext}"
    if input_video.resolve() != dst.resolve():
        shutil.copy2(str(input_video), str(dst))
    return dst


def pick_speaker_box(boxes: List[Box]) -> Optional[Box]:
    greens = [b for b in boxes if b.color == "green"]
    if not greens:
        return None
    # pick the biggest green box (usually the speaker)
    return max(greens, key=lambda b: b.area)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, default=".", help="Path to TalkNet-ASD repo root")
    p.add_argument("--video", type=str, required=True, help="Path to input video (.mp4/.avi)")
    p.add_argument("--video_name", type=str, required=True, help="Name to use with demoTalkNet.py (e.g., 001)")
    p.add_argument("--timestamp", type=float, required=True, help="Timestamp in seconds (e.g., 12.34)")
    p.add_argument("--force_rerun", action="store_true", help="Re-run TalkNet even if output exists")
    p.add_argument("--out_dir", type=str, default="out", help="Output folder")
    p.add_argument("--confidence_threshold", type=float, default=-0.5, help="Confidence threshold for active speaker detection (lower = more permissive, default: -0.5)")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_video = Path(args.video).resolve()
    video_name = args.video_name
    ts = float(args.timestamp)

    if not (repo_root / "demoTalkNet.py").exists():
        raise FileNotFoundError(
            f"demoTalkNet.py not found in {repo_root}. "
            f"Run this script from the TalkNet-ASD repo root or pass --repo_root."
        )

    # 1) Copy video into demo/ so TalkNet demo can find it
    demo_video = ensure_video_in_demo(repo_root, input_video, video_name)
    print(f"[Prep] Video ready at: {demo_video}")

    # 2) Run TalkNet demo (produces annotated output)
    annotated_video = run_talknet_demo(repo_root, video_name, force=args.force_rerun, confidence_threshold=args.confidence_threshold)
    print(f"[TalkNet] Annotated output: {annotated_video}  (green=active, red=non-active)")

    # 3) Read frame at timestamp from both original and annotated (should align)
    cap_orig = open_video(demo_video)
    cap_ann = open_video(annotated_video)

    orig_frame, fidx = read_frame_at_timestamp(cap_orig, ts)
    ann_frame, _ = read_frame_at_timestamp(cap_ann, ts)

    cap_orig.release()
    cap_ann.release()

    # 4) Detect boxes (faces) from annotated frame
    boxes = find_colored_boxes(ann_frame)
    speaker_box = pick_speaker_box(boxes)

    # 5) Draw overlay on the ORIGINAL frame for a clean screenshot
    drawn = draw_boxes(orig_frame, boxes, speaker_box)

    # 6) Save outputs
    ms = int(round(ts * 1000))
    out_dir = Path(args.out_dir) / f"{video_name}_t{ms}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save drawn frame
    drawn_path = out_dir / "frame_with_boxes.jpg"
    cv2.imwrite(str(drawn_path), drawn)
    print(f"[OK] Saved annotated screenshot: {drawn_path}")

    # Save raw original frame too
    raw_path = out_dir / "frame_raw.jpg"
    cv2.imwrite(str(raw_path), orig_frame)
    print(f"[OK] Saved raw screenshot: {raw_path}")

    # Save crops
    save_outputs(out_dir, orig_frame, boxes, speaker_box)

    # Print a compact result
    if speaker_box is None:
        print("[Result] Speaker: NOT FOUND (no green box detected at this timestamp)")
    else:
        print(
            f"[Result] Speaker box (green): x1={speaker_box.x1},y1={speaker_box.y1},x2={speaker_box.x2},y2={speaker_box.y2}"
        )
    print(f"[Result] Total face boxes detected: {len(boxes)}")


if __name__ == "__main__":
    main()
