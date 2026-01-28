"""SQLite database for person onboarding."""
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid

# Database file path - stored at memobot root level
MEMOBOT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = MEMOBOT_ROOT / "persons.db"


def get_db_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_database():
    """Initialize the database and create the persons table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            face_id TEXT NOT NULL UNIQUE,
            speaker_id TEXT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for faster lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_id ON persons(face_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_speaker_id ON persons(speaker_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON persons(name)")
    
    conn.commit()
    conn.close()
    print(f"[Database] Initialized database at {DB_PATH}")


def add_person(face_id: str, name: str, speaker_id: Optional[str] = None) -> str:
    """
    Add a new person to the database.
    
    Args:
        face_id: The face identifier (UUID, used as filename in face_database)
        name: The person's name
        speaker_id: Optional speaker identifier (can be set later)
    
    Returns:
        The generated person_id
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Generate unique person_id
    person_id = str(uuid.uuid4())
    
    try:
        cursor.execute("""
            INSERT INTO persons (person_id, face_id, speaker_id, name, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (person_id, face_id, speaker_id, name))
        
        conn.commit()
        print(f"[Database] Added person: person_id={person_id}, face_id={face_id}, name={name}")
        return person_id
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed: persons.face_id" in str(e):
            print(f"[Database] Person with face_id '{face_id}' already exists")
            # Get existing person_id
            cursor.execute("SELECT person_id FROM persons WHERE face_id = ?", (face_id,))
            result = cursor.fetchone()
            if result:
                return result[0]
        raise
    finally:
        conn.close()


def update_speaker_id(person_id: str, speaker_id: str):
    """Update the speaker_id for a person."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE persons 
        SET speaker_id = ?, updated_at = CURRENT_TIMESTAMP
        WHERE person_id = ?
    """, (speaker_id, person_id))
    
    conn.commit()
    conn.close()
    print(f"[Database] Updated speaker_id for person_id={person_id} to {speaker_id}")


def get_person_by_face_id(face_id: str) -> Optional[Dict[str, Any]]:
    """Get a person by their face_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM persons WHERE face_id = ?", (face_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_person_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get a person by their name."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM persons WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_person_by_speaker_id(speaker_id: str) -> Optional[Dict[str, Any]]:
    """Get a person by their speaker_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM persons WHERE speaker_id = ?", (speaker_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_all_persons() -> List[Dict[str, Any]]:
    """Get all persons from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM persons ORDER BY created_at")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("[Database] Database initialized successfully")
