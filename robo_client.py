import requests
import json
import os
import sys
import subprocess
import asyncio
import base64
from dotenv import load_dotenv

# Load env from this repo root explicitly (not cwd-dependent)
DOTENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# Audio constants for Realtime API (24kHz, 16-bit PCM, mono)
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_DURATION_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class RealtimeAgent:
    """Agent using OpenAI's Realtime API for audio-to-audio interaction."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.ws = None
        self.audio_queue = asyncio.Queue()
        self.is_recording = False
        self.input_stream = None
        self.output_stream = None
        self.sd = None  # sounddevice module
        
    async def get_ephemeral_token(self):
        """Get an ephemeral client secret for WebSocket connection."""
        session_config = {
            "model": "gpt-4o-realtime-preview-2024-12-17",
            "voice": "verse",
            "instructions": "You are a helpful assistant with access to a memory system. When users ask about past events, use the retrieveMemory function to find relevant information. When you are unsure, do not make up information. Keep responses concise for voice interaction.",
            "tools": [
                {
                    "type": "function",
                    "name": "retrieveMemory",
                    "description": "Retrieve past memories based on a natural language query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queryText": {
                                "type": "string",
                                "description": "The natural language query to search for memories. The query should be specific about the event or item being searched.",
                            },
                        },
                        "required": ["queryText"],
                    },
                }
            ],
            "tool_choice": "auto",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
        }
        
        response = requests.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=session_config,
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get ephemeral token: {response.status_code} {response.text}")
        
        data = response.json()
        return data["client_secret"]["value"]
    
    def retrieve_memory(self, query):
        """Retrieve memories using the existing pipeline."""
        twelvelabs_dir = os.path.join(os.path.dirname(__file__), 'examples', 'twelvelabs')
        venv_python = os.path.join(twelvelabs_dir, 'venv', 'bin', 'python')
        query_file = os.path.join(twelvelabs_dir, 'query.py')
        
        if not os.path.exists(query_file):
            return {"events": [], "metadata": {"error": "query.py not found"}}
        
        if not os.path.exists(venv_python):
            venv_python = sys.executable
        
        try:
            escaped_query = query.replace('"', '\\"').replace("'", "\\'")
            
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
    
except Exception as e:
    error_msg = {{
        "events": [],
        "metadata": {{"error": f"Error: {{str(e)}}"}}
    }}
    print(json.dumps(error_msg))
'''],
                capture_output=True,
                text=True,
                cwd=twelvelabs_dir
            )
            
            if result.returncode != 0:
                return {"events": [], "metadata": {"error": result.stderr}}
            
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]
            memory_context = json.loads(json_line)
            
            if memory_context.get("events"):
                print(f"\n📋 Retrieved {len(memory_context['events'])} relevant memories")
            
            return memory_context
            
        except Exception as e:
            return {"events": [], "metadata": {"error": str(e)}}
    
    async def connect(self):
        """Connect to the Realtime API via WebSocket."""
        try:
            import websockets
        except ImportError:
            print("Please install websockets: pip install websockets")
            sys.exit(1)
        
        print("🔑 Getting ephemeral token...")
        token = await self.get_ephemeral_token()
        
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        headers = {
            "Authorization": f"Bearer {token}",
            "OpenAI-Beta": "realtime=v1",
        }
        
        print("🔌 Connecting to Realtime API...")
        self.ws = await websockets.connect(url, additional_headers=headers)
        print("✅ Connected!")
        
        return self.ws
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to the API."""
        if self.ws:
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }))
    
    async def commit_audio(self):
        """Commit the audio buffer and request a response."""
        if self.ws:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
            }))
            await self.ws.send(json.dumps({
                "type": "response.create",
            }))
    
    async def handle_function_call(self, call_id: str, name: str, arguments: str):
        """Handle function calls from the API."""
        try:
            args = json.loads(arguments)
            
            if name == "retrieveMemory":
                print(f"\n🔍 Searching memories for: {args.get('queryText', '')}")
                result = self.retrieve_memory(args.get("queryText", ""))
                
                await self.ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    }
                }))
                
                await self.ws.send(json.dumps({
                    "type": "response.create",
                }))
                
        except Exception as e:
            print(f"Error handling function call: {e}")
    
    async def receive_messages(self):
        """Receive and process messages from the API."""
        try:
            import sounddevice as sd
            import numpy as np
            self.sd = sd
        except ImportError:
            print("Please install sounddevice and numpy: pip install sounddevice numpy")
            sys.exit(1)
        
        # Create output stream for playback
        self.output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=CHUNK_SIZE,
        )
        self.output_stream.start()
        
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type", "")
                
                if event_type == "session.created":
                    print("🎙️  Session started. Speak into your microphone...")
                    print("    Press Ctrl+C to stop.\n")
                
                elif event_type == "input_audio_buffer.speech_started":
                    print("🎤 Speech detected...")
                
                elif event_type == "input_audio_buffer.speech_stopped":
                    print("🛑 Speech ended, processing...")
                
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if transcript:
                        print(f"\n[You] {transcript}")
                
                elif event_type == "response.audio.delta":
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        # Convert bytes to numpy array for sounddevice
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        self.output_stream.write(audio_array)
                
                elif event_type == "response.audio_transcript.delta":
                    transcript = event.get("delta", "")
                    if transcript:
                        print(transcript, end="", flush=True)
                
                elif event_type == "response.audio_transcript.done":
                    print()  # New line after transcript
                
                elif event_type == "response.function_call_arguments.done":
                    call_id = event.get("call_id", "")
                    name = event.get("name", "")
                    arguments = event.get("arguments", "{}")
                    await self.handle_function_call(call_id, name, arguments)
                
                elif event_type == "response.done":
                    response = event.get("response", {})
                    if response.get("status") == "failed":
                        error = response.get("status_details", {}).get("error", {})
                        print(f"\n❌ Error: {error.get('message', 'Unknown error')}")
                
                elif event_type == "error":
                    error = event.get("error", {})
                    print(f"\n❌ API Error: {error.get('message', 'Unknown error')}")
                    
        except Exception as e:
            print(f"\nConnection error: {e}")
    
    async def capture_audio(self):
        """Capture audio from microphone and send to API."""
        try:
            import sounddevice as sd
            import numpy as np
            self.sd = sd
        except ImportError:
            print("Please install sounddevice and numpy: pip install sounddevice numpy")
            sys.exit(1)
        
        self.is_recording = True
        loop = asyncio.get_event_loop()
        
        def audio_callback(indata, frames, time_info, status):
            """Called for each audio block from the microphone."""
            if status:
                print(f"Audio status: {status}")
            if self.is_recording:
                # Convert numpy array to bytes
                audio_bytes = indata.tobytes()
                # Schedule coroutine on the event loop
                asyncio.run_coroutine_threadsafe(self.send_audio(audio_bytes), loop)
        
        try:
            # Open input stream with callback
            self.input_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='int16',
                blocksize=CHUNK_SIZE,
                callback=audio_callback,
            )
            self.input_stream.start()
            
            # Keep running while recording
            while self.is_recording:
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"Audio capture error: {e}")
        finally:
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
    
    async def run(self):
        """Main run loop for the realtime agent."""
        try:
            await self.connect()
            
            # Run audio capture and message receiving concurrently
            await asyncio.gather(
                self.capture_audio(),
                self.receive_messages(),
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.is_recording = False
        
        if self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except:
                pass
        
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except:
                pass


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


def run_realtime_mode():
    """Run the agent in realtime audio mode."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY is not set.")
        print(f"Looked for .env at: {DOTENV_PATH}")
        print("\nAdd to .env:")
        print("  OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    if not api_key.startswith("sk-"):
        print("\nERROR: OPENAI_API_KEY does not look like an OpenAI key.")
        sys.exit(1)
    
    print("\n🎙️  Memobot Realtime Audio Mode")
    print("=" * 40)
    print("This mode uses OpenAI's Realtime API for")
    print("voice-to-voice conversation with memory access.")
    print("\nRequirements:")
    print("  - sounddevice (pip install sounddevice)")
    print("  - numpy (pip install numpy)")
    print("  - websockets (pip install websockets)")
    print("=" * 40)
    
    agent = RealtimeAgent(api_key)
    asyncio.run(agent.run())


def run_text_mode():
    """Run the agent in text mode (original behavior)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("\nERROR: OPENROUTER_API_KEY is not set.")
        print(f"Looked for .env at: {DOTENV_PATH}")
        print("\nAdd to .env:")
        print("  OPENROUTER_API_KEY=sk-or-v1-...")
        sys.exit(1)

    if not api_key.startswith("sk-or-"):
        print("\nERROR: OPENROUTER_API_KEY does not look like an OpenRouter key (expected prefix sk-or-).")
        sys.exit(1)

    memobot_url = os.environ.get("MEMOBOT_API_URL", "http://localhost:8000")
    agent = Agent(api_key, memobot_api_url=memobot_url)

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
            
            print()
            response = agent.chat(user_input)
            print(f"\n[Agent] {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memobot - AI assistant with memory")
    parser.add_argument(
        "--mode", 
        choices=["text", "realtime", "audio"],
        default="text",
        help="Interaction mode: 'text' for chat, 'realtime' or 'audio' for voice"
    )
    
    args = parser.parse_args()
    
    if args.mode in ["realtime", "audio"]:
        run_realtime_mode()
    else:
        run_text_mode()
