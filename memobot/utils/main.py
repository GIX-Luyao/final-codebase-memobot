#!/usr/bin/env python3
"""
Create new person records from images in the new_users folder.

This script:
1. Scans the new_users folder for image files
2. Extracts the filename (without extension) as the person's name
3. Generates a random UUID for face_id
4. Copies images to the face_database folder with UUID as filename
5. Creates database records with person_id, face_id (UUID), name (speaker_id is NULL initially)
6. Generates DeepFace embeddings and updates the embedding cache (skips if already exists)

Usage:
    python create_new_person.py
"""

import sys
import os
import uuid
from pathlib import Path
import shutil

# Add parent directory to path to import deepface utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import deepface utilities
import importlib.util
deepface_path = Path(__file__).parent.parent / "deepface" / "match_face.py"
spec = importlib.util.spec_from_file_location("match_face", deepface_path)
match_face = importlib.util.module_from_spec(spec)
spec.loader.exec_module(match_face)

IMAGE_EXTS = match_face.IMAGE_EXTS
set_mac_stability_env = match_face.set_mac_stability_env
get_embedding = match_face.get_embedding
l2_normalize = match_face.l2_normalize
load_db_cache = match_face.load_db_cache
build_db_cache = match_face.build_db_cache
list_images = match_face.list_images

from utils.database import init_database, add_person, get_person_by_name

# Set Mac stability environment variables
set_mac_stability_env(max_threads=1)

# Face matching configuration (matching ingest_pipeline settings)
FACE_MODEL = "ArcFace"
FACE_DETECTOR = "opencv"
FACE_DISTANCE_METRIC = "cosine"
FACE_ENFORCE_DETECTION = False
FACE_ALIGN = False

# Paths (repo root = parent of package for face_database)
UTILS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = UTILS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
NEW_USERS_DIR = UTILS_DIR / "new_users"
FACE_DATABASE_DIR = REPO_ROOT / "face_database"
FACE_CACHE_PATH = PACKAGE_ROOT / "deepface" / "db_embeddings.pkl"


def get_image_files(directory: Path) -> list[Path]:
    """Get all image files from a directory."""
    if not directory.exists():
        print(f"[Warning] Directory does not exist: {directory}")
        return []
    
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
            image_files.append(file_path)
    
    return sorted(image_files)


def update_face_embedding_cache(face_id: str, image_path: Path) -> bool:
    """
    Update the face embedding cache with a new image.
    
    Args:
        face_id: The face_id (UUID) to use as the key
        image_path: Path to the image file
    
    Returns:
        True if embedding was generated and cached, False if skipped or failed
    """
    import pickle
    import numpy as np
    
    # Check if embedding already exists in cache
    if FACE_CACHE_PATH.exists():
        try:
            cached = load_db_cache(FACE_CACHE_PATH)
            # Check if config matches
            if (
                cached.get("model_name") == FACE_MODEL
                and cached.get("detector_backend") == FACE_DETECTOR
                and cached.get("distance_metric") == FACE_DISTANCE_METRIC
                and cached.get("align") == FACE_ALIGN
                and cached.get("enforce_detection") == FACE_ENFORCE_DETECTION
            ):
                db_map = cached.get("db_map", {})
                # Check if this face_id already has an embedding
                if face_id in db_map:
                    print(f"  [Skip] Embedding already exists in cache for face_id: {face_id}")
                    return False
                
                # Generate embedding for new image
                print(f"  [Embedding] Generating face embedding...")
                emb = get_embedding(
                    img_path=image_path,
                    model_name=FACE_MODEL,
                    detector_backend=FACE_DETECTOR,
                    enforce_detection=FACE_ENFORCE_DETECTION,
                    align=FACE_ALIGN,
                )
                
                if emb is None:
                    print(f"  [Warning] Could not generate embedding (no face detected)")
                    return False
                
                # Normalize if using cosine distance
                if FACE_DISTANCE_METRIC == "cosine":
                    emb = l2_normalize(emb)
                
                # Add to cache
                db_map[face_id] = {
                    "embedding": emb,
                    "db_image": str(image_path),
                }
                
                # Ensure cache directory exists
                FACE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                
                # Save updated cache
                with open(FACE_CACHE_PATH, "wb") as f:
                    pickle.dump(
                        {
                            "model_name": FACE_MODEL,
                            "detector_backend": FACE_DETECTOR,
                            "distance_metric": FACE_DISTANCE_METRIC,
                            "align": FACE_ALIGN,
                            "enforce_detection": FACE_ENFORCE_DETECTION,
                            "db_map": db_map,
                        },
                        f,
                    )
                
                print(f"  [OK] Embedding generated and cached")
                return True
        except Exception as e:
            print(f"  [Warning] Error loading cache, will rebuild: {e}")
    
    # Cache doesn't exist or config mismatch - rebuild entire cache
    print(f"  [Embedding] Rebuilding face embedding cache...")
    try:
        # Ensure cache directory exists
        FACE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        db_images = list_images(FACE_DATABASE_DIR)
        if not db_images:
            print(f"  [Warning] No images found in face_database, cannot build cache")
            return False
        
        db_map = build_db_cache(
            db_images=db_images,
            cache_path=FACE_CACHE_PATH,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=FACE_ENFORCE_DETECTION,
            align=FACE_ALIGN,
            distance_metric=FACE_DISTANCE_METRIC,
        )
        
        if face_id in db_map:
            print(f"  [OK] Embedding generated and cached (full rebuild)")
            return True
        else:
            print(f"  [Warning] Embedding not found after rebuild (face may not have been detected)")
            return False
    except Exception as e:
        print(f"  [Error] Failed to rebuild cache: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_new_users():
    """Process all images in the new_users folder and create person records."""
    # Initialize database
    print("=== Initializing Database ===")
    init_database()
    
    # Ensure face_database directory exists
    FACE_DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Face database directory: {FACE_DATABASE_DIR}")
    
    # Get all image files from new_users folder
    print(f"\n=== Scanning {NEW_USERS_DIR} for images ===")
    image_files = get_image_files(NEW_USERS_DIR)
    
    if not image_files:
        print(f"[Warning] No image files found in {NEW_USERS_DIR}")
        print(f"[Info] Supported formats: {', '.join(IMAGE_EXTS)}")
        return
    
    print(f"[Info] Found {len(image_files)} image file(s)")
    
    # Process each image
    print(f"\n=== Processing Images ===")
    processed_count = 0
    skipped_count = 0
    
    for image_path in image_files:
        # Extract name from filename (without extension)
        name = image_path.stem
        
        print(f"\n--- Processing: {image_path.name} ---")
        print(f"  Name: {name}")
        
        # Check if person with this name already exists
        existing = get_person_by_name(name)
        if existing:
            print(f"  [Skip] Person with name '{name}' already exists in database")
            print(f"         Existing person_id: {existing['person_id']}")
            print(f"         Existing face_id: {existing['face_id']}")
            skipped_count += 1
            continue
        
        # Generate random UUID for face_id
        face_id = str(uuid.uuid4())
        print(f"  Face ID: {face_id}")
        
        # Copy image to face_database folder with UUID as filename (preserve extension)
        dest_filename = f"{face_id}{image_path.suffix}"
        dest_path = FACE_DATABASE_DIR / dest_filename
        
        if dest_path.exists():
            print(f"  [Warning] Image already exists in face_database: {dest_filename}")
            print(f"  [Info] Skipping copy, but will still create database record")
        else:
            try:
                shutil.copy2(image_path, dest_path)
                print(f"  [OK] Copied to face_database: {dest_filename}")
            except Exception as e:
                print(f"  [Error] Failed to copy image: {e}")
                continue
        
        # Add person to database
        try:
            person_id = add_person(face_id=face_id, name=name, speaker_id=None)
            print(f"  [OK] Created database record: person_id={person_id}")
        except Exception as e:
            print(f"  [Error] Failed to create database record: {e}")
            # Remove copied image if database insert failed
            if dest_path.exists():
                try:
                    dest_path.unlink()
                    print(f"  [Info] Removed copied image due to database error")
                except:
                    pass
            continue
        
        # Generate and cache face embedding
        print(f"\n  === Generating Face Embedding ===")
        embedding_success = update_face_embedding_cache(face_id, dest_path)
        if embedding_success:
            processed_count += 1
        else:
            print(f"  [Warning] Person added but embedding generation failed or skipped")
            processed_count += 1  # Still count as processed since DB record was created
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"  Processed: {processed_count} person(s)")
    print(f"  Skipped: {skipped_count} person(s)")
    print(f"  Total images: {len(image_files)}")
    
    if processed_count > 0:
        from utils.database import DB_PATH
        print(f"\n[Success] {processed_count} new person(s) added to database")
        print(f"[Info] Database location: {DB_PATH}")
        print(f"[Info] Face images copied to: {FACE_DATABASE_DIR}")
        print(f"[Info] Face embeddings cache: {FACE_CACHE_PATH}")


if __name__ == "__main__":
    try:
        process_new_users()
    except KeyboardInterrupt:
        print("\n[Interrupted] Process cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
