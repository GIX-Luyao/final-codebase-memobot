import os
import json
import base64
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load env vars
load_dotenv()

class ActionManager:
    def __init__(self, repo_name: Optional[str] = None, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.repo_name = repo_name or os.getenv("GITHUB_REPO")
        
        if not self.token or not self.repo_name:
            raise ValueError("GitHub Token and Repo Name must be provided via env vars or args.")
            
        self.base_url = f"https://api.github.com/repos/{self.repo_name}"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        self.index_path = "index.json"
        self._index_cache = None

    def _request(self, method: str, path: str, data: Optional[Dict] = None, params: Optional[Dict] = None):
        url = f"{self.base_url}/{path}"
        # For testing purposes, mock the request if in test environment and no side_effect set
        # But properly we should rely on unittest.mock to handle this.
        
        response = requests.request(method, url, headers=self.headers, json=data, params=params)
        
        # In test_core.py, the side_effect logic for status_code doesn't automatically prevent raise_for_status 
        # from raising if we returned a real Response object with 401.
        # But we are using MagicMock. 
        # The issue is likely that self.mock_requests.request return value is not being used correctly.
        
        if response.status_code >= 400 and response.status_code != 404:
             # Let 404 be handled by caller
             response.raise_for_status()
             
        return response

    def _get_index(self, force_refresh=False) -> List[Dict]:
        if self._index_cache is not None and not force_refresh:
            return self._index_cache

        response = self._request("GET", f"contents/{self.index_path}")
        if response.status_code == 404:
            self._index_cache = []
        else:
            content_data = response.json()
            json_content = base64.b64decode(content_data['content']).decode('utf-8')
            self._index_cache = json.loads(json_content)
            
        return self._index_cache

    def _save_index(self, index_data: List[Dict], message: str):
        content_json = json.dumps(index_data, indent=2)
        encoded_content = base64.b64encode(content_json.encode('utf-8')).decode('utf-8')
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if exists to get sha
                response = self._request("GET", f"contents/{self.index_path}")
                sha = response.json().get('sha') if response.status_code == 200 else None
                
                payload = {
                    "message": message,
                    "content": encoded_content
                }
                if sha:
                    payload["sha"] = sha
                    
                self._request("PUT", f"contents/{self.index_path}", data=payload)
                
                # Update cache
                self._index_cache = index_data
                return # Success
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 409 and attempt < max_retries - 1:
                    print(f"Conflict updating index (409), retrying... ({attempt+1}/{max_retries})")
                    import time
                    time.sleep(1)
                    continue
                raise e

    def save_action(self, name: str, code: str, keywords: List[str], message: str) -> Dict[str, Any]:
        """
        Save action code and update index.
        """
        file_path = f"actions/{name}.py"
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
        
        # 1. Update/Create Code File
        response = self._request("GET", f"contents/{file_path}")
        sha = response.json().get('sha') if response.status_code == 200 else None
        
        payload = {
            "message": f"Update {name}: {message}",
            "content": encoded_code
        }
        if sha:
            payload["sha"] = sha
            
        self._request("PUT", f"contents/{file_path}", data=payload)
        
        # Update Index
        index = self._get_index(force_refresh=True)
        
        # Check if exists
        existing_item = next((item for item in index if item["name"] == name), None)
        
        # Generate ID if new
        import uuid
        action_id = existing_item["id"] if existing_item else str(uuid.uuid4())
        
        new_item = {
            "id": action_id,
            "name": name,
            "keywords": keywords,
            "updated_at": str(datetime.now()),
            "message": message,
            "path": file_path
        }
        
        if existing_item:
            # Update existing
            index = [item if item["name"] != name else new_item for item in index]
        else:
            # Add new
            index.append(new_item)
            
        self._save_index(index, f"Update index for {name}")
        
        return new_item

    def search_actions(self, query: str) -> List[Dict[str, Any]]:
        """
        Search in local cached index. Fast and free.
        """
        index = self._get_index()
        results = []
        
        query = query.lower()
        
        for item in index:
            # Fuzzy match in name or keywords
            in_name = query in item["name"].lower()
            in_keywords = any(query in k.lower() for k in item["keywords"])
            
            if in_name or in_keywords:
                # Fetch code content for the result
                try:
                    code_content = self.get_action_code(item["name"])
                    item["code"] = code_content
                except Exception:
                    item["code"] = ""
                results.append(item)
                
        return results

    def get_action_code(self, name: str, version_sha: Optional[str] = None) -> str:
        """
        Fetch the actual code from GitHub.
        """
        path = f"actions/{name}.py"
        params = {"ref": version_sha} if version_sha else None
        
        response = self._request("GET", f"contents/{path}", params=params)
        if response.status_code == 200:
            return base64.b64decode(response.json()['content']).decode('utf-8')
        return ""

    def list_actions(self) -> List[str]:
        index = self._get_index()
        return [item["name"] for item in index]

    def rollback_action(self, name: str, steps: int = 1) -> Dict[str, Any]:
        """
        Rollback by finding old commit and reapplying it.
        """
        path = f"actions/{name}.py"
        
        # Get history
        response = self._request("GET", f"commits", params={"path": path})
        if response.status_code != 200:
             raise ValueError("Failed to fetch history")
             
        commits = response.json()
        
        # We need steps + 1 (0 is current)
        if len(commits) <= steps:
             raise ValueError("Not enough history")
             
        target_commit = commits[steps]
        target_sha = target_commit['sha']
        
        # Get content at that commit
        old_code = self.get_action_code(name, version_sha=target_sha)
        
        if not old_code:
            raise ValueError(f"Could not fetch code for commit {target_sha}")
        
        # Save as new version
        self.save_action(name, old_code, [], f"Rollback to {target_sha[:7]}")
        
        return {"status": "success", "code": old_code, "version": target_sha}

    def delete_action(self, name: str, message: str = "Delete action") -> bool:
        """
        Delete an action:
        1. Remove the code file from GitHub
        2. Remove the entry from index.json
        """
        file_path = f"actions/{name}.py"
        
        # 1. Delete File from GitHub
        try:
            # Get current SHA to delete
            response = self._request("GET", f"contents/{file_path}")
            if response.status_code == 200:
                sha = response.json().get('sha')
                self._request("DELETE", f"contents/{file_path}", data={
                    "message": f"Delete {name}: {message}",
                    "sha": sha
                })
            elif response.status_code != 404:
                # If error other than 404, raise it
                response.raise_for_status()
        except Exception as e:
            # If deletion fails (e.g. file not found), we still want to clean up the index
            print(f"Warning: Failed to delete file {file_path}: {e}")

        # 2. Update Index
        index = self._get_index(force_refresh=True)
        
        # Filter out the deleted item
        new_index = [item for item in index if item["name"] != name]
        
        if len(new_index) != len(index):
            self._save_index(new_index, f"Remove {name} from index")
            return True
            
        return False
