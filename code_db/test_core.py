import unittest
from unittest.mock import MagicMock, patch
import os
import requests
import json
import base64
import sys

# Add parent directory to path so we can import code_db
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_db.core import ActionManager

# Flag to choose between real integration test or mock test
# Set to True to test against real GitHub repo (requires .env)
USE_REAL_GITHUB = True

class TestActionManager(unittest.TestCase):
    def setUp(self):
        if USE_REAL_GITHUB:
            # Check if env vars are present
            if not os.getenv("GITHUB_TOKEN") or not os.getenv("GITHUB_REPO"):
                self.skipTest("GITHUB_TOKEN or GITHUB_REPO not set in environment")
            self.manager = ActionManager()
            
            # Use a unique test prefix to avoid collision
            import time
            self.test_prefix = f"test_action_{int(time.time())}"
        else:
            # Mock setup (existing code)
            self.patcher = patch('code_db.core.requests.request')
            self.mock_request = self.patcher.start()
            self.manager = ActionManager(token="fake", repo_name="user/repo")

    def tearDown(self):
        if not USE_REAL_GITHUB:
            self.patcher.stop()

    def test_save_and_search_real(self):
        if not USE_REAL_GITHUB:
            return

        print(f"\n=== Testing Real GitHub Integration (Population & Search) ===")
        
        # Define a set of meaningful actions without random suffixes in names
        # We will rely on overwriting existing ones for this test, which is fine.
        actions_to_create = [
            {
                "name": "dance_disco",
                "code": "def dance():\n    print('Dancing disco!')\n    move_arms('up')\n    move_legs('step')",
                "keywords": ["dance", "party", "fun", "disco"],
                "message": "Add disco dance move"
            },
            {
                "name": "kick_football",
                "code": "def kick():\n    print('Kicking football!')\n    balance('left_leg')\n    swing('right_leg')",
                "keywords": ["sport", "kick", "ball", "football"],
                "message": "Add football kick"
            },
            {
                "name": "wave_hand",
                "code": "def wave():\n    print('Waving hand!')\n    raise_arm('right')\n    wave_wrist()",
                "keywords": ["greeting", "hello", "wave"],
                "message": "Add hand wave"
            }
        ]

        # 1. Save Actions
        print("\n1. Populating Actions...")
        for action in actions_to_create:
            print(f"   Saving {action['name']}...")
            self.manager.save_action(
                name=action['name'],
                code=action['code'],
                keywords=action['keywords'],
                message=action['message']
            )
        print("   Population complete.")

        # 2. List All Actions
        print("\n2. Listing All Actions...")
        all_actions = self.manager.list_actions()
        print(f"   Total actions in DB: {len(all_actions)}")
        for name in all_actions:
            print(f"   - {name}")
        
        # Verify our created actions are in the list
        for action in actions_to_create:
            self.assertIn(action['name'], all_actions)

        # 3. Search Tests
        print("\n3. Testing Search...")
        
        search_queries = [
            ("dance", 1),      # Should find dance_disco
            ("ball", 1),       # Should find kick_football
            ("greeting", 1)    # Should find wave_hand
        ]
        
        for query, expected_min_count in search_queries:
            print(f"   Searching for '{query}'...")
            results = self.manager.search_actions(query)
            print(f"   Found {len(results)} results.")
            for r in results:
                print(f"     -> {r['name']} (ID: {r.get('id', 'N/A')})")
            
            found_ours = any(r["name"] in [a['name'] for a in actions_to_create] for r in results)
            self.assertTrue(found_ours, f"Expected to find our created action for query '{query}'")

        # 4. Get Code Test
        print("\n4. Fetching Code for 'kick_football'...")
        target_name = "kick_football"
        code = self.manager.get_action_code(target_name)
        print("   Code content:")
        print(code)
        self.assertIn("Kicking football", code)

        # 5. Delete Action Test
        print("\n5. Deleting Action 'kick_football'...")
        deleted = self.manager.delete_action("kick_football")
        self.assertTrue(deleted, "Expected delete_action to return True")
        
        # Verify it's gone from list
        all_actions_after = self.manager.list_actions()
        self.assertNotIn("kick_football", all_actions_after)
        
        # Verify it's gone from search
        results = self.manager.search_actions("ball")
        found_deleted = any(r["name"] == "kick_football" for r in results)
        self.assertFalse(found_deleted, "Deleted action should not be found in search")
        print("   Deletion verified.")

    def test_mock_save_action(self):
        if USE_REAL_GITHUB:
            return
            
        # ... existing mock test code ...
        def request_side_effect(method, url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock() 
            
            if "actions/test.py" in url and method == "GET":
                mock_resp.status_code = 404
            elif "index.json" in url and method == "GET":
                mock_resp.status_code = 404
            elif method == "PUT":
                mock_resp.status_code = 201
                mock_resp.json.return_value = {"content": {"sha": "new_sha"}}
            else:
                mock_resp.status_code = 200
            return mock_resp
            
        self.mock_request.side_effect = request_side_effect
        
        self.manager.save_action("test", "print('hi')", ["tag"], "msg")
        
        # Verify calls
        put_calls = [call for call in self.mock_request.call_args_list if call[0][0] == 'PUT']
        self.assertEqual(len(put_calls), 2)


if __name__ == "__main__":
    unittest.main()
