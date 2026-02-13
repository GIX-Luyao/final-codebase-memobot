# Code DB

This module handles storage and version control for robot actions.

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

# Search actions
results = manager.search_actions("dance")
for action in results:
    print(action["name"], action["version_hash"])

# Rollback
manager.rollback_action("disco_move", steps=1)
```

## Testing

Run the tests:
```bash
python3 code_db/test_core.py
```
