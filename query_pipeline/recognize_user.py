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

# Paths
QUERY_PIPELINE_DIR = Path(__file__).parent
IMAGE_PATH = QUERY_PIPELINE_DIR / "image.png"
FACE_DATABASE_DIR = QUERY_PIPELINE_DIR.parent / "face_database"
FACE_CACHE_PATH = QUERY_PIPELINE_DIR.parent / "deepface" / "db_embeddings.pkl"

# Set Mac stability environment variables
match_face.set_mac_stability_env(max_threads=1)


def recognize_user(image_path=None):
    """Recognize a user from an image and output their database record.

    Args:
        image_path: Path to the query image. Defaults to query_pipeline/image.png.
    """
    path = Path(image_path) if image_path else IMAGE_PATH
    if not path.exists():
        print(f"[Error] Image not found: {path}")
        print(f"[Info] Please place an image in the query_pipeline directory or pass a valid path")
        sys.exit(1)

    print(f"[Info] Loading image: {path}")
    
    # Ensure database is initialized
    init_database()
    
    # Load or build face database cache
    print(f"[Info] Loading face database from: {FACE_DATABASE_DIR}")
    
    if not FACE_DATABASE_DIR.exists():
        print(f"[Error] Face database directory not found: {FACE_DATABASE_DIR}")
        sys.exit(1)
    
    # Build/load DB embeddings cache
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
        
        if not db_map:
            print(f"[Error] No valid face embeddings in database")
            print(f"[Info] Try running onboarding/create_new_person.py first")
            sys.exit(1)
        
        print(f"[Info] Face database loaded: {len(db_map)} person(s)")
        
    except Exception as e:
        print(f"[Error] Failed to load face database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Match the query image to the database
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
        )
        
        if face_id is None:
            print(f"[Error] No face detected in image or no match found")
            sys.exit(1)
        
        print(f"[Match] Found match: face_id={face_id}, distance={distance:.4f}")
        print(f"[Match] Matched database image: {db_image}")
        
    except Exception as e:
        print(f"[Error] Face matching failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Query the database for person information
    print(f"\n[Info] Querying database for face_id: {face_id}")
    try:
        person = get_person_by_face_id(face_id)
        
        if person is None:
            print(f"[Error] Person not found in database for face_id: {face_id}")
            print(f"[Info] The face was matched but no database record exists")
            sys.exit(1)
        
        # Output the person record
        print(f"\n=== Person Record ===")
        print(f"person_id: {person['person_id']}")
        print(f"face_id: {person['face_id']}")
        print(f"name: {person['name']}")
        print(f"speaker_id: {person['speaker_id'] if person['speaker_id'] else 'None'}")
        print(f"created_at: {person['created_at']}")
        print(f"updated_at: {person['updated_at']}")
        
        # Also output as a dictionary for programmatic use
        print(f"\n=== Full Record (dict) ===")
        print(person)
        
        return person
        
    except Exception as e:
        print(f"[Error] Database query failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        recognize_user()
    except KeyboardInterrupt:
        print("\n[Interrupted] Process cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
