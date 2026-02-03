"""SQLite database for person onboarding."""
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid

# Database file path - stored at repo root (parent of package)
MEMOBOT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = MEMOBOT_ROOT / "persons.db"


def get_db_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_database(verbose=True):
    """Initialize the database and create the persons table if it doesn't exist.
    Schema has no speaker_id; identity is by face_id only.
    verbose: If False, suppress log messages (e.g. when used from realtime speaker-ID)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            face_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrate existing table: drop speaker_id if present (SQLite < 3.35 has no DROP COLUMN)
    cursor.execute("PRAGMA table_info(persons)")
    columns = [row[1] for row in cursor.fetchall()]
    if "speaker_id" in columns:
        cursor.execute("""
            CREATE TABLE persons_new (
                person_id TEXT PRIMARY KEY,
                face_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO persons_new (person_id, face_id, name, created_at, updated_at)
            SELECT person_id, face_id, name, created_at, updated_at FROM persons
        """)
        cursor.execute("DROP TABLE persons")
        cursor.execute("ALTER TABLE persons_new RENAME TO persons")
        if verbose:
            print("[Database] Migrated persons table: removed speaker_id column")

    # Create indexes for faster lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_id ON persons(face_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON persons(name)")

    conn.commit()
    conn.close()
    if verbose:
        print(f"[Database] Initialized database at {DB_PATH}")


def add_person(face_id: str, name: str) -> str:
    """
    Add a new person to the database.

    Args:
        face_id: The face identifier (UUID, used as filename in face_database)
        name: The person's name

    Returns:
        The generated person_id
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Generate unique person_id
    person_id = str(uuid.uuid4())

    try:
        cursor.execute("""
            INSERT INTO persons (person_id, face_id, name, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (person_id, face_id, name))
        
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
