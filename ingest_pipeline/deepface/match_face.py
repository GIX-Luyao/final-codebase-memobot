#!/usr/bin/env python3
"""
match_face.py

Mac-friendly face matching using DeepFace:
- Precompute database embeddings once and cache to disk
- Compare query embeddings to database embeddings (fast)
- person_id = database filename (stem)

Why this works better on mac:
- Avoids DeepFace.find() rescanning DB for each query
- Limits native thread explosion (OpenMP/BLAS) that can cause mutex blocking logs
- Defaults to lightweight detector backend (opencv)

Usage:
  python3 match_face.py --queries_dir ./detected_faces --db_dir ./face_database --out_csv ./results.csv
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from deepface import DeepFace

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -------------------------
# Mac stability knobs
# -------------------------
def set_mac_stability_env(max_threads: int = 1) -> None:
    """
    Reduce the chance of native-thread contention / mutex blocking on macOS.
    Call before heavy TF/OpenCV work starts.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduce TF logging noise
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def person_id_from_path(img_path: Path) -> str:
    return img_path.stem  # filename without extension


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 1 - cosine similarity
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - (np.dot(a, b) / denom))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n


def get_embedding(
    img_path: Path,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
) -> Optional[np.ndarray]:
    """
    Returns a single face embedding vector or None if no face found (when enforce_detection=False).
    If multiple faces are present, DeepFace.represent returns a list; we take the first face.
    """
    try:
        reps = DeepFace.represent(
            img_path=str(img_path),
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
        if not reps:
            return None
        # DeepFace returns list[dict], each dict has "embedding"
        emb = reps[0].get("embedding", None)
        if emb is None:
            return None
        return np.array(emb, dtype=np.float32)
    except Exception:
        return None


def build_db_cache(
    db_images: List[Path],
    cache_path: Path,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    distance_metric: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Builds embeddings for each DB image and caches:
    {
      person_id: {
         "embedding": np.ndarray (normalized if needed),
         "db_image": str path
      }, ...
    }

    If there are multiple images per person_id, keep the best-quality one by default:
    - First valid embedding wins (simple & predictable).
    """
    db_map: Dict[str, Dict[str, np.ndarray]] = {}

    for img in db_images:
        pid = person_id_from_path(img)
        if pid in db_map:
            continue  # keep first; customize if you want multi-photo per person

        emb = get_embedding(
            img_path=img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
        )
        if emb is None:
            continue

        # For cosine, L2 normalize embeddings for stable comparisons
        if distance_metric == "cosine":
            emb = l2_normalize(emb)

        db_map[pid] = {
            "embedding": emb,
            "db_image": str(img),
        }

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "model_name": model_name,
                "detector_backend": detector_backend,
                "distance_metric": distance_metric,
                "align": align,
                "enforce_detection": enforce_detection,
                "db_map": db_map,
            },
            f,
        )

    return db_map


def load_db_cache(cache_path: Path) -> Dict:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def ensure_db_cache(
    db_dir: Path,
    cache_path: Path,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    distance_metric: str,
    rebuild: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    db_images = list_images(db_dir)
    if not db_images:
        raise RuntimeError(f"No images found in db_dir: {db_dir}")

    if cache_path.exists() and not rebuild:
        cached = load_db_cache(cache_path)
        # If config changed, rebuild
        if (
            cached.get("model_name") == model_name
            and cached.get("detector_backend") == detector_backend
            and cached.get("distance_metric") == distance_metric
            and cached.get("align") == align
            and cached.get("enforce_detection") == enforce_detection
        ):
            return cached["db_map"]

    return build_db_cache(
        db_images=db_images,
        cache_path=cache_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        distance_metric=distance_metric,
    )


def match_query_to_db(
    query_img: Path,
    db_map: Dict[str, Dict[str, np.ndarray]],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    distance_metric: str,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Returns (best_person_id, best_db_image_path, best_distance)
    """
    q_emb = get_embedding(
        img_path=query_img,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    if q_emb is None:
        return None, None, None

    if distance_metric == "cosine":
        q_emb = l2_normalize(q_emb)

    best_pid = None
    best_db_img = None
    best_dist = None

    for pid, info in db_map.items():
        d_emb = info["embedding"]

        if distance_metric == "cosine":
            dist = cosine_distance(q_emb, d_emb)
        elif distance_metric == "euclidean":
            dist = euclidean_distance(q_emb, d_emb)
        elif distance_metric == "euclidean_l2":
            dist = euclidean_distance(l2_normalize(q_emb), l2_normalize(d_emb))
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pid = pid
            best_db_img = info["db_image"]

    return best_pid, best_db_img, best_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries_dir", required=True, help="Folder with query face images")
    ap.add_argument("--db_dir", required=True, help="Folder with database images (filename=person_id)")
    ap.add_argument("--out_csv", default="results.csv", help="Output CSV path")
    ap.add_argument("--cache_path", default="db_embeddings.pkl", help="Cache file for DB embeddings")

    # Mac-safe defaults
    ap.add_argument("--model_name", default="ArcFace",
                    choices=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"])
    ap.add_argument("--detector_backend", default="opencv",
                    choices=["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "yunet", "fastmtcnn", "centerface"])
    ap.add_argument("--distance_metric", default="cosine",
                    choices=["cosine", "euclidean", "euclidean_l2"])

    ap.add_argument("--enforce_detection", action="store_true",
                    help="If set, error when no face is detected. Default: False (skip).")
    ap.add_argument("--align", action="store_true",
                    help="If set, align faces. Default: False (faster on mac).")
    ap.add_argument("--rebuild_cache", action="store_true", help="Force rebuild DB cache")
    ap.add_argument("--max_threads", type=int, default=1, help="Limit native threads (mac stability). Default: 1")

    args = ap.parse_args()

    set_mac_stability_env(max_threads=args.max_threads)

    queries_dir = Path(args.queries_dir)
    db_dir = Path(args.db_dir)
    out_csv = Path(args.out_csv)
    cache_path = Path(args.cache_path)

    query_images = list_images(queries_dir)
    if not query_images:
        raise RuntimeError(f"No images found in queries_dir: {queries_dir}")

    # Build/load DB embeddings cache
    db_map = ensure_db_cache(
        db_dir=db_dir,
        cache_path=cache_path,
        model_name=args.model_name,
        detector_backend=args.detector_backend,
        enforce_detection=args.enforce_detection,
        align=args.align,
        distance_metric=args.distance_metric,
        rebuild=args.rebuild_cache,
    )

    if not db_map:
        raise RuntimeError("No valid face embeddings in DB. (Try a different detector_backend or use clearer images.)")

    print(f"[Info] Queries: {len(query_images)} images")
    print(f"[Info] DB (embeddings): {len(db_map)} persons")
    print(f"[Info] model={args.model_name}, detector={args.detector_backend}, metric={args.distance_metric}, threads={args.max_threads}")
    print(f"[Info] cache={cache_path.resolve()}")

    rows = []
    for qi in query_images:
        pid, db_img, dist = match_query_to_db(
            query_img=qi,
            db_map=db_map,
            model_name=args.model_name,
            detector_backend=args.detector_backend,
            enforce_detection=args.enforce_detection,
            align=args.align,
            distance_metric=args.distance_metric,
        )

        rows.append({
            "query_image": str(qi),
            "person_id": pid,
            "db_image": db_img,
            "distance": dist,
            "matched": pid is not None,
        })

        print(f"[Match] {qi.name} -> {pid} (dist={dist})")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[Done] Wrote: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
