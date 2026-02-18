# Code DB

This module handles storage and version control for robot actions (GitHub-backed index and `actions/*.py` files).

## Setup

Set in your environment (or `.env`):

- `GITHUB_TOKEN` – token with repo read/write
- `GITHUB_REPO` – repo in `owner/repo` form

## Usage

```python
from code_db import ActionManager

manager = ActionManager()

# Save an action
manager.save_action(
    name="disco_move",
    code="print('dance')",
    keywords=["dance", "fun"],
    message="Initial version"
)

# Search actions (matches name and keywords)
results = manager.search_actions("dance")
for action in results:
    print(action["name"], action.get("keywords"), action.get("path"))

# Get action code
code = manager.get_action_code("disco_move")

# Rollback
manager.rollback_action("disco_move", steps=1)
```

## Memobot integration

The Gemini Live client (`memobot/query_pipeline/gemini_client.py`) uses code_db when `GITHUB_TOKEN` and `GITHUB_REPO` are set:

- **searchRobotActions** – calls `ActionManager().search_actions(query)` (and optional category filter).
- **saveCode** – calls `ActionManager().save_action(...)`; if the user doesn’t pass code, the last generated/run code from **writeCode** is saved. Filename is used as the action name (e.g. `wave_hand` or `wave_hand.py`).

See the tool descriptions and `_get_action_manager_instance()` in `gemini_client.py` for details.

## Testing

Run the tests:

```bash
python3 code_db/test_core.py
```
