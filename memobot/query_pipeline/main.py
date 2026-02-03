#!/usr/bin/env python3
"""
main.py

Entry point: load an image, recognize the user via face matching, then run
the robot client with that person's name in the system prompt.
"""

import sys
from pathlib import Path

# Ensure project root is on path for recognize_user's imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recognize_user import recognize_user
from robo_client import run_realtime_mode


def main():
    image_path = Path(__file__).parent / "image.png"
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

    print(f"[main] Recognizing user from image: {image_path}")
    person = recognize_user(image_path=str(image_path))
    name = person["name"]
    person_id = person.get("person_id")
    print(f"[main] Recognized: {name} (person_id={person_id}). Starting robo_client.\n")
    run_realtime_mode(user_name=name, person_id=person_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[main] Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n[main] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
