import requests
import json
import os
import sys
import subprocess
from dotenv import load_dotenv

# Load env from this repo root explicitly (not cwd-dependent)
DOTENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

class Agent:
    def __init__(self, api_key, memobot_api_url="http://localhost:8000"):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to a memory system. When users ask about past events, use the retrieveMemory function to find relevant information. When you are unsure, do not make up information.",
            }
        ]

    def _validate_openrouter_key(self) -> bool:
        """
        Validates the OpenRouter key by calling /auth/key.
        This avoids debugging chat failures when the key/account is invalid.
        """
        try:
            r = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            if r.status_code != 200:
                print(f"[Agent] OpenRouter key validation failed: HTTP {r.status_code}")
                print(f"[Agent] Body: {r.text}")
                return False
            data = r.json()
            # Response typically includes data like: {"data": {"label": ..., "usage": ..., ...}}
            print("[Agent] OpenRouter key validation OK")
            return True
        except Exception as e:
            print(f"[Agent] OpenRouter key validation error: {e}")
            return False

    def retrieve_memory(self, query):
        print(f'[Memory API] Retrieving memory for query: "{query}"')
        
        # Call the Twelve Labs retrieval pipeline via subprocess
        twelvelabs_dir = os.path.join(os.path.dirname(__file__), 'twelvelabs')
        venv_python = os.path.join(twelvelabs_dir, 'venv', 'bin', 'python')
        
        # Check if venv exists
        if not os.path.exists(venv_python):
            error_msg = f"""
[Memory API] ERROR: Virtual environment not found!

Please set up the Twelve Labs environment first:

    cd {twelvelabs_dir}
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Then run this script again.
"""
            print(error_msg)
            return {"events": [], "metadata": {"error": "Virtual environment not set up. See console for instructions."}}
        
        try:
            # Escape quotes in query for shell command
            escaped_query = query.replace('"', '\\"').replace("'", "\\'")
            
            # Run query.py in the virtual environment and capture JSON output
            result = subprocess.run(
                [venv_python, '-c', f'''
import sys
sys.path.insert(0, "{twelvelabs_dir}")
from query import retrieve_and_rank
import json

results = retrieve_and_rank(
    question="""{escaped_query}""",
    index_name="twelve-labs",
    top_k=10,
    alpha=0.5,
    beta=0.3,
    gamma=0.2,
)

# Transform results into the expected format
events = []
for r in results:
    md = r.get("metadata", {{}})
    event = {{
        "id": r["id"],
        "summary": md.get("summary", ""),
        "video_file": md.get("video_file", ""),
        "start_time_sec": md.get("start_time_sec"),
        "end_time_sec": md.get("end_time_sec"),
        "timestamp_utc": md.get("timestamp_utc"),
        "importance_score": md.get("importance_score"),
        "talking_to_camera": md.get("talking_to_camera"),
        "relevance_score": r["relevance_score"],
        "final_score": r["final_score"],
    }}
    events.append(event)

memory_context = {{
    "events": events,
    "metadata": {{
        "total_results": len(events),
        "query": """{escaped_query}""",
    }}
}}

print(json.dumps(memory_context))
'''],
                capture_output=True,
                text=True,
                cwd=twelvelabs_dir
            )
            
            if result.returncode != 0:
                print(f"[Memory API] Error running query: {result.stderr}")
                return {"events": [], "metadata": {"error": result.stderr}}
            
            # Parse the JSON output (filter out any debug prints)
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]  # Last line should be the JSON
            memory_context = json.loads(json_line)
            print(f'[Memory API] Found {len(memory_context.get("events", []))} relevant memory(ies)')
            return memory_context
            
        except Exception as e:
            print(f"[Memory API] Error calling retrieval pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {"events": [], "metadata": {"error": str(e)}}

    def reset_conversation(self):
        """Reset the conversation history to start fresh."""
        print('[Agent] Resetting conversation history')
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to a memory system. When users ask about past events, use the retrieveMemory function to find relevant information. When you are unsure, do not make up information.",
            }
        ]

    def chat(self, user_query):
        print(f'[Agent] Received user query: "{user_query}"')
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieveMemory",
                    "description": "Retrieve past memories based on a natural language query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queryText": {
                                "type": "string",
                                "description": "The natural language query to search for memories. The query should be specific about the event or item being searched. Specific details such as time, location and name/person_id will help retrieve the most relevant memories.",
                            },
                        },
                        "required": ["queryText"],
                    },
                },
            }
        ]

        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_query,
        })

        try:
            print(f'[Agent] Calling GPT API with tools...')
            # First API call to get tool usage
            response = requests.post(
                url=self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "openai/gpt-4o-mini",
                    "messages": self.conversation_history,
                    "tools": tools,
                    "tool_choice": "auto",
                }),
            )

            # Check if request was successful
            if response.status_code != 200:
                print(f"[Agent] API Error: Status {response.status_code}")
                print(f"[Agent] Response: {response.text}")
                error_message = f"API returned error status {response.status_code}"
                self.conversation_history.append({
                    "role": "assistant",
                    "content": error_message,
                })
                return error_message

            data = response.json()
            
            # Check if response has expected structure
            if "choices" not in data:
                print(f"[Agent] Unexpected API response structure:")
                print(json.dumps(data, indent=2))
                error_message = "Unexpected API response format"
                self.conversation_history.append({
                    "role": "assistant",
                    "content": error_message,
                })
                return error_message

            assistant_message = data["choices"][0]["message"]

            # Check if the model wants to call a function
            if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                tool_call = assistant_message["tool_calls"][0]
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                print(f'[Agent] GPT wants to call function: {function_name}')
                print(f'[Agent] Function arguments: {function_args}')

                if function_name == "retrieveMemory":
                    # Execute the memory retrieval
                    memory_context = self.retrieve_memory(function_args["queryText"])

                    print(f'[Agent] Memory context retrieved:')
                    print(json.dumps(memory_context, indent=2))

                    # Add the function result to conversation history
                    self.conversation_history.append(assistant_message)
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(memory_context),
                    })

                    print(f'[Agent] Calling GPT API with memory context...')
                    # Second API call with function result
                    final_response = requests.post(
                        url=self.api_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps({
                            "model": "openai/gpt-4o-mini",
                            "messages": self.conversation_history,
                        }),
                    )

                    if final_response.status_code != 200:
                        print(f"[Agent] API Error on second call: Status {final_response.status_code}")
                        print(f"[Agent] Response: {final_response.text}")
                        error_message = f"API returned error status {final_response.status_code}"
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": error_message,
                        })
                        return error_message

                    final_data = final_response.json()
                    final_message = final_data["choices"][0]["message"]["content"]
                    print(f'[Agent] Final response generated')
                    
                    # Add assistant's final response to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_message,
                    })
                    
                    return final_message

            # If no tool call, add assistant message to history and return
            content = assistant_message.get("content", "")
            self.conversation_history.append({
                "role": "assistant",
                "content": content,
            })
            return content

        except Exception as error:
            print(f"[Agent] Error calling OpenRouter API: {error}")
            import traceback
            traceback.print_exc()
            error_message = "Sorry, I encountered an error while processing your request."
            self.conversation_history.append({
                "role": "assistant",
                "content": error_message,
            })
            return error_message


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Memobot Agent Test")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("\nERROR: OPENROUTER_API_KEY is not set.")
        print(f"Looked for .env at: {DOTENV_PATH}")
        print("\nAdd to .env:")
        print("  OPENROUTER_API_KEY=sk-or-v1-...")
        sys.exit(1)

    # Sanity checks + diagnostics (do not print full key)
    redacted = api_key[:10] + "..." + api_key[-6:] if len(api_key) > 20 else "(too short)"
    print(f"[Agent] Loaded .env from: {DOTENV_PATH if os.path.exists(DOTENV_PATH) else '(missing)'}")
    print(f"[Agent] Using OPENROUTER_API_KEY={redacted}")

    # Quick format sanity check (doesn't guarantee validity, but catches obvious mistakes)
    if not api_key.startswith("sk-or-"):
        print("\nERROR: OPENROUTER_API_KEY does not look like an OpenRouter key (expected prefix sk-or-).")
        sys.exit(1)

    memobot_url = os.environ.get("MEMOBOT_API_URL", "http://localhost:8000")
    agent = Agent(api_key, memobot_api_url=memobot_url)

    # Fail fast if the key/account is invalid (your current error indicates this)
    if not agent._validate_openrouter_key():
        print("\nERROR: OpenRouter rejected this API key (e.g. 'User not found').")
        print("Fix: generate a new key at https://openrouter.ai/keys and update OPENROUTER_API_KEY.")
        sys.exit(1)

    print("\nWelcome to Memobot! Type 'exit' or 'quit' to end the conversation.")
    print("Type 'reset' to start a new conversation.\n")
    
    while True:
        try:
            user_input = input("[You] ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                print("\nConversation reset. Starting fresh!\n")
                continue
            
            print()  # Empty line for spacing
            response = agent.chat(user_input)
            print(f"\n[Agent] {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
    
    print("=" * 60)
