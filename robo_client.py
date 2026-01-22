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
        self.last_retrieved_memories = []  # Keep track of last retrieved memories
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
                print(f"OpenRouter key validation failed: HTTP {r.status_code}")
                return False
            return True
        except Exception as e:
            print(f"OpenRouter key validation error: {e}")
            return False

    def retrieve_memory(self, query):
        # Call the Twelve Labs retrieval pipeline via subprocess
        twelvelabs_dir = os.path.join(os.path.dirname(__file__), 'examples', 'twelvelabs')
        venv_python = os.path.join(twelvelabs_dir, 'venv', 'bin', 'python')
        query_file = os.path.join(twelvelabs_dir, 'query.py')
        
        # Check if the query.py file exists
        if not os.path.exists(query_file):
            print(f"ERROR: query.py not found at {query_file}")
            print("Please ensure the Twelve Labs example files are in the correct location.")
            return {"events": [], "metadata": {"error": "query.py not found"}}
        
        # Check if venv exists, if not try system python as fallback
        if not os.path.exists(venv_python):
            print("Virtual environment not found, trying system python...")
            venv_python = sys.executable  # Use current Python interpreter
        
        try:
            # Escape quotes in query for shell command
            escaped_query = query.replace('"', '\\"').replace("'", "\\'")
            
            # Run query.py in the virtual environment and capture JSON output
            result = subprocess.run(
                [venv_python, '-c', f'''
import sys
sys.path.insert(0, "{twelvelabs_dir}")
try:
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
    
except ImportError as e:
    error_msg = {{
        "events": [],
        "metadata": {{
            "error": f"Missing dependencies: {{str(e)}}. Please install required packages.",
            "setup_instructions": [
                "cd {twelvelabs_dir}",
                "python3 -m venv venv",
                "source venv/bin/activate",
                "pip install -r requirements.txt"
            ]
        }}
    }}
    print(json.dumps(error_msg))
except Exception as e:
    error_msg = {{
        "events": [],
        "metadata": {{
            "error": f"Error in query execution: {{str(e)}}"
        }}
    }}
    print(json.dumps(error_msg))
'''],
                capture_output=True,
                text=True,
                cwd=twelvelabs_dir
            )
            
            if result.returncode != 0:
                print(f"Error running memory query: {result.stderr}")
                return {"events": [], "metadata": {"error": result.stderr}}
            
            # Parse the JSON output (filter out any debug prints)
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]  # Last line should be the JSON
            memory_context = json.loads(json_line)
            
            # Check if there was an error in the response
            if memory_context.get("metadata", {}).get("error"):
                error = memory_context["metadata"]["error"]
                setup_instructions = memory_context["metadata"].get("setup_instructions", [])
                print(f"Memory system error: {error}")
                if setup_instructions:
                    print("\nSetup instructions:")
                    for instruction in setup_instructions:
                        print(f"  {instruction}")
                return memory_context
            
            # Store retrieved memories
            self.last_retrieved_memories = memory_context.get("events", [])
            
            # Display what was retrieved
            if self.last_retrieved_memories:
                print(f"\n📋 Retrieved {len(self.last_retrieved_memories)} relevant memories:")
                for i, event in enumerate(self.last_retrieved_memories[:3], 1):  # Show first 3
                    print(f"  {i}. {event.get('summary', 'No summary')[:100]}...")
                if len(self.last_retrieved_memories) > 3:
                    print(f"  ... and {len(self.last_retrieved_memories) - 3} more\n")
            else:
                print("📋 No relevant memories found\n")
            
            return memory_context
            
        except Exception as e:
            print(f"Error calling retrieval pipeline: {e}")
            return {"events": [], "metadata": {"error": str(e)}}

    def get_last_memories(self):
        """Return the last retrieved memories."""
        return self.last_retrieved_memories

    def reset_conversation(self):
        """Reset the conversation history to start fresh."""
        self.last_retrieved_memories = []  # Clear retrieved memories too
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to a memory system. When users ask about past events, use the retrieveMemory function to find relevant information. When you are unsure, do not make up information.",
            }
        ]

    def chat(self, user_query):
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
                print(f"API Error: Status {response.status_code}")
                error_message = f"API returned error status {response.status_code}"
                self.conversation_history.append({
                    "role": "assistant",
                    "content": error_message,
                })
                return error_message

            data = response.json()
            
            # Check if response has expected structure
            if "choices" not in data:
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

                if function_name == "retrieveMemory":
                    # Execute the memory retrieval
                    memory_context = self.retrieve_memory(function_args["queryText"])

                    # Add the function result to conversation history
                    self.conversation_history.append(assistant_message)
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(memory_context),
                    })

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
                        print(f"API Error on second call: Status {final_response.status_code}")
                        error_message = f"API returned error status {final_response.status_code}"
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": error_message,
                        })
                        return error_message

                    final_data = final_response.json()
                    final_message = final_data["choices"][0]["message"]["content"]
                    
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
            print(f"Error calling OpenRouter API: {error}")
            error_message = "Sorry, I encountered an error while processing your request."
            self.conversation_history.append({
                "role": "assistant",
                "content": error_message,
            })
            return error_message


# Example usage
if __name__ == "__main__":
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("\nERROR: OPENROUTER_API_KEY is not set.")
        print(f"Looked for .env at: {DOTENV_PATH}")
        print("\nAdd to .env:")
        print("  OPENROUTER_API_KEY=sk-or-v1-...")
        sys.exit(1)

    # Quick format sanity check (doesn't guarantee validity, but catches obvious mistakes)
    if not api_key.startswith("sk-or-"):
        print("\nERROR: OPENROUTER_API_KEY does not look like an OpenRouter key (expected prefix sk-or-).")
        sys.exit(1)

    memobot_url = os.environ.get("MEMOBOT_API_URL", "http://localhost:8000")
    agent = Agent(api_key, memobot_api_url=memobot_url)

    # Fail fast if the key/account is invalid
    if not agent._validate_openrouter_key():
        print("\nERROR: OpenRouter rejected this API key.")
        print("Fix: generate a new key at https://openrouter.ai/keys and update OPENROUTER_API_KEY.")
        sys.exit(1)

    print("\nWelcome to Memobot! Type 'exit' or 'quit' to end the conversation.")
    print("Type 'reset' to start a new conversation.")
    print("Type 'memories' to see the last retrieved memories.\n")
    
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
            
            if user_input.lower() == 'memories':
                memories = agent.get_last_memories()
                if memories:
                    print(f"\n📋 Last retrieved memories ({len(memories)} total):")
                    for i, event in enumerate(memories, 1):
                        print(f"\n{i}. Summary: {event.get('summary', 'No summary')}")
                        if event.get('video_file'):
                            print(f"   Video: {event.get('video_file')}")
                        if event.get('timestamp_utc'):
                            print(f"   Time: {event.get('timestamp_utc')}")
                        print(f"   Relevance: {event.get('relevance_score', 'N/A')}")
                else:
                    print("\n📋 No memories retrieved yet")
                print()
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
