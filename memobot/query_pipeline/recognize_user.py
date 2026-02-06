#!/usr/bin/env python3
"""
recognize_user.py

Recognizes a person from an image by:
1. Loading image.png from the query_pipeline directory
2. Matching the face against the face_database
3. Querying the SQL database for the person's information
4. Outputting the person record (person_id, name, etc.)

Usage:
    python recognize_user.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import deepface utilities
import importlib.util
deepface_path = Path(__file__).parent.parent / "deepface" / "match_face.py"
spec = importlib.util.spec_from_file_location("match_face", deepface_path)
match_face = importlib.util.module_from_spec(spec)
spec.loader.exec_module(match_face)

# Import database utilities
from utils.database import get_person_by_face_id, init_database

# Face matching configuration (matching ingest_pipeline settings)
FACE_MODEL = "ArcFace"
FACE_DETECTOR = "opencv"
FACE_DISTANCE_METRIC = "cosine"
FACE_ENFORCE_DETECTION = False
FACE_ALIGN = False

# Paths (package root = query_pipeline's parent; repo root = one level up for face_database/persons.db)
QUERY_PIPELINE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = QUERY_PIPELINE_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
IMAGE_PATH = QUERY_PIPELINE_DIR / "image.png"
FACE_DATABASE_DIR = REPO_ROOT / "face_database"
FACE_CACHE_PATH = PACKAGE_ROOT / "deepface" / "db_embeddings.pkl"

# Set Mac stability environment variables
match_face.set_mac_stability_env(max_threads=1)


def recognize_user(image_path=None, verbose=True):
    """Recognize a user from an image and output their database record.

    Args:
        image_path: Path to the query image. Defaults to query_pipeline/image.png.
        verbose: If True, print [Info], [Timing], [Match], and person record. If False, only errors (for speaker-ID in realtime).
    """
    path = Path(image_path) if image_path else IMAGE_PATH
    if not path.exists():
        if verbose:
            print(f"[Error] Image not found: {path}")
        return None

    if verbose:
        print(f"[Info] Loading image: {path}")

    # Ensure database is initialized
    init_database(verbose=verbose)

    if verbose:
        print(f"[Info] Loading face database from: {FACE_DATABASE_DIR}")

    if not FACE_DATABASE_DIR.exists():
        if verbose:
            print(f"[Error] Face database directory not found: {FACE_DATABASE_DIR}")
        return None

    # Build/load DB embeddings cache (allow empty db for library callers that will register unknown users)
    try:
        db_map = match_face.ensure_db_cache(
            db_dir=FACE_DATABASE_DIR,
            cache_path=FACE_CACHE_PATH,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=FACE_ENFORCE_DETECTION,
            align=FACE_ALIGN,
            distance_metric=FACE_DISTANCE_METRIC,
            rebuild=False,
        )
    except Exception as e:
        # Empty face_database or load error: if query has a face, caller can register
        if verbose:
            print(f"[Info] Face database not ready ({e}); checking if query image has a face")
        q_emb = match_face.get_embedding(
            img_path=path,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=FACE_ENFORCE_DETECTION,
            align=FACE_ALIGN,
        )
        return {"unknown": True} if q_emb is not None else None

    if not db_map:
        if verbose:
            print(f"[Info] No face embeddings in database; checking if query image has a face")
        q_emb = match_face.get_embedding(
            img_path=path,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=FACE_ENFORCE_DETECTION,
            align=FACE_ALIGN,
        )
        return {"unknown": True} if q_emb is not None else None

    if verbose:
        print(f"[Info] Face database loaded: {len(db_map)} person(s)")

    # Match the query image to the database
    if verbose:
        print(f"\n[Info] Matching face in image...")
    try:
        face_id, db_image, distance = match_face.match_query_to_db(
            query_img=path,
            db_map=db_map,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=FACE_ENFORCE_DETECTION,
            align=FACE_ALIGN,
            distance_metric=FACE_DISTANCE_METRIC,
            verbose=verbose,
        )

        if face_id is None:
            # Face may be detected but not matched to anyone; allow caller to register as new user
            q_emb = match_face.get_embedding(
                img_path=path,
                model_name=FACE_MODEL,
                detector_backend=FACE_DETECTOR,
                enforce_detection=FACE_ENFORCE_DETECTION,
                align=FACE_ALIGN,
            )
            if verbose and q_emb is None:
                print(f"[Error] No face detected in image")
            return {"unknown": True} if q_emb is not None else None

        if verbose:
            print(f"[Match] Found match: face_id={face_id}, distance={distance:.4f}")
            print(f"[Match] Matched database image: {db_image}")

    except Exception as e:
        if verbose:
            print(f"[Error] Face matching failed: {e}")
            import traceback
            traceback.print_exc()
        return None

    # Query the database for person information
    if verbose:
        print(f"\n[Info] Querying database for face_id: {face_id}")
    try:
        person = get_person_by_face_id(face_id)

        if person is None:
            if verbose:
                print(f"[Error] Person not found in database for face_id: {face_id}")
            return None

        # Output the person record (only when verbose)
        if verbose:
            print(f"\n=== Person Record ===")
            print(f"person_id: {person['person_id']}")
            print(f"face_id: {person['face_id']}")
            print(f"name: {person['name']}")
            print(f"created_at: {person['created_at']}")
            print(f"updated_at: {person['updated_at']}")
            print(f"\n=== Full Record (dict) ===")
            print(person)
        # Include distance for confidence checks (lower = better for cosine)
        person["distance"] = distance
        return person

    except Exception as e:
        if verbose:
            print(f"[Error] Database query failed: {e}")
            import traceback
            traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        r = recognize_user()
        if r is None or (isinstance(r, dict) and r.get("unknown")):
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted] Process cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
