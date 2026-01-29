import requests
import json
import os
import sys
import asyncio
import base64
from dotenv import load_dotenv

from agent import Agent

# Load env from repo root (parent of query_pipeline)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOTENV_PATH = os.path.join(_REPO_ROOT, ".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# Try to import MemobotService
MemobotService = None

# Method 1: Try direct import (if installed as package)
try:
    from Memobot import MemobotService
    print("[Info] Memobot package loaded successfully (installed package)")
except ImportError as e1:
    # Method 2: Try importing from root directory
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    try:
        from Memobot import MemobotService
        print("[Info] Memobot package loaded successfully (from repo root)")
    except ImportError as e2:
        print(f"[Warning] Memobot package not found")
        print(f"   - Direct import error: {e1}")
        print(f"   - Repo root import error: {e2}")
        print(f"   - Repo root path: {_REPO_ROOT}")
        
        # Check if Memobot directory exists
        memobot_dir = os.path.join(_REPO_ROOT, "Memobot")
        if os.path.exists(memobot_dir):
            print(f"   - Memobot directory exists at: {memobot_dir}")
            print(f"   - Contents: {os.listdir(memobot_dir)}")
            init_file = os.path.join(memobot_dir, "__init__.py")
            if os.path.exists(init_file):
                print(f"   - __init__.py exists")
            else:
                print(f"   - __init__.py NOT FOUND - this may be the issue!")
        else:
            print(f"   - Memobot directory NOT FOUND at: {memobot_dir}")
        
        print("\n[Info] To install Memobot as a package, run one of:")
        print(f"   cd {_REPO_ROOT} && pip install -e ./Memobot")
        print("   or")
        print(f"   pip install -e {memobot_dir}")
        print("\n[Warning] Knowledge Graph retrieval will be skipped.")

# Audio constants for Realtime API (24kHz, 16-bit PCM, mono)
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_DURATION_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class RealtimeAgent:
    """Agent using OpenAI's Realtime API for audio-to-audio interaction."""
    
    def __init__(self, api_key, user_name=None, person_id=None):
        self.api_key = api_key
        self.user_name = user_name
        self.person_id = person_id  # Direct person_id for knowledge graph queries
        self.ws = None
        self.audio_queue = asyncio.Queue()
        self.is_recording = False
        self.input_stream = None
        self.output_stream = None
        self.sd = None  # sounddevice module
        self.memobot_service = None  # Knowledge graph service
        self._init_memobot_service()
    
    def _init_memobot_service(self):
        """Initialize the MemobotService for knowledge graph queries."""
        if MemobotService is None:
            return
        try:
            self.memobot_service = MemobotService.from_env(group_id='tenant_001')
            print("[Info] MemobotService initialized for knowledge graph retrieval")
        except Exception as e:
            print(f"[Warning] Failed to initialize MemobotService: {e}")
            self.memobot_service = None
        
    async def get_ephemeral_token(self):
        """Get an ephemeral client secret for WebSocket connection."""
        base_instructions = "You are a helpful assistant with access to a memory system. You must speak english. When users ask about past events, use the retrieveMemory function to find relevant information. When you are unsure, do not make up information. Keep responses concise for voice interaction."
        if self.user_name:
            instructions = f"You are speaking with {self.user_name}. " + base_instructions
        else:
            instructions = base_instructions
        session_config = {
            "model": "gpt-4o-realtime-preview-2024-12-17",
            "voice": "verse",
            "instructions": instructions,
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
    
    async def retrieve_memory(self, query):
        """Retrieve memories using both vector DB and knowledge graph."""
        vector_results = []
        graph_results = []
        
        # 1. Query Vector Database (existing logic)
        try:
            from query import retrieve_and_rank
            
            print(f"[DEBUG] Querying vector DB with: {query}")
            
            results = retrieve_and_rank(
                question=query,
                index_name="twelve-labs",
                top_k=10,
                alpha=0.5,
                beta=0.3,
                gamma=0.2,
            )
            
            for r in results:
                md = r.get("metadata", {})
                event = {
                    "id": r["id"],
                    "source": "vector_db",
                    "summary": md.get("summary", ""),
                    "video_file": md.get("video_file", ""),
                    "start_time_sec": md.get("start_time_sec"),
                    "end_time_sec": md.get("end_time_sec"),
                    "timestamp_utc": md.get("timestamp_utc"),
                    "importance_score": md.get("importance_score"),
                    "relevance_score": r["relevance_score"],
                    "final_score": r["final_score"],
                }
                vector_results.append(event)
                
        except ImportError as e:
            print(f"[DEBUG] Vector DB import error: {e}")
        except Exception as e:
            import traceback
            print(f"[DEBUG] Vector DB query error: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        # 2. Query Knowledge Graph (now using await directly)
        if self.memobot_service:
            try:
                print(f"[DEBUG] Querying knowledge graph with: {query}")
                graph_results = await self._query_knowledge_graph(query)
            except Exception as e:
                import traceback
                print(f"[DEBUG] Knowledge graph query error: {e}")
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        # 3. Merge and format results
        all_events = vector_results + graph_results
        
        memory_context = {
            "events": all_events,
            "metadata": {
                "total_results": len(all_events),
                "vector_db_results": len(vector_results),
                "graph_db_results": len(graph_results),
                "query": query,
            }
        }
        
        # Print results summary
        if all_events:
            print(f"\n📋 Retrieved {len(all_events)} memories ({len(vector_results)} from vector DB, {len(graph_results)} from knowledge graph):")
            for i, event in enumerate(all_events[:5], 1):
                source = event.get('source', 'unknown')
                print(f"\n  [{i}] Source: {source} | ID: {event.get('id', 'N/A')}")
                summary = event.get('summary') or event.get('text', 'N/A')
                print(f"      Content: {summary[:150]}..." if len(str(summary)) > 150 else f"      Content: {summary}")
                if event.get('person_name'):
                    print(f"      Person: {event.get('person_name')}")
                if event.get('video_file'):
                    print(f"      Video: {event.get('video_file')}")
            if len(all_events) > 5:
                print(f"\n  ... and {len(all_events) - 5} more memories")
            print()
        else:
            print(f"[DEBUG] No events found from either source")
        
        return memory_context

    async def _query_knowledge_graph(self, query: str) -> list[dict]:
        """Query the knowledge graph for relevant memories."""
        results = []
        
        if not self.memobot_service:
            return results
        
        try:
            # Determine person_id for the query
            person_id = self.person_id  # Use directly provided person_id first
            
            if not person_id and self.user_name:
                # Map user_name to person_id convention: user_{lowercase_name}_001
                person_id = f"user_{self.user_name.lower()}_001"
                print(f"[DEBUG] Mapped user_name '{self.user_name}' to person_id '{person_id}'")
            
            if not person_id:
                print(f"[DEBUG] No person_id available, knowledge graph query requires person_id")
                return results
            
            print(f"[DEBUG] Querying knowledge graph with person_id: {person_id}")
            
            graph_response = await self.memobot_service.retrieve(
                query,
                person_id=person_id,
            )
            
            # Parse the response structure from MemobotService
            # Response format: {'center_person': {...}, 'events': [...], ...}
            if graph_response:
                if isinstance(graph_response, dict):
                    # Extract events from the response
                    events = graph_response.get("events", [])
                    center_person = graph_response.get("center_person", {})
                    
                    # If no events but we have center_person info, create a result from it
                    if not events and center_person:
                        results.append({
                            "id": center_person.get("uuid", "kg_person"),
                            "source": "knowledge_graph",
                            "person_id": center_person.get("person_id", ""),
                            "person_name": center_person.get("name", ""),
                            "text": f"Person: {center_person.get('name', 'Unknown')}",
                            "timestamp_utc": center_person.get("updated_at", ""),
                        })
                    
                    # Process events
                    for event in events:
                        if isinstance(event, dict):
                            results.append({
                                "id": event.get("uuid", event.get("id", "")),
                                "source": "knowledge_graph",
                                "person_id": event.get("person_id", ""),
                                "person_name": event.get("person_name", center_person.get("name", "")),
                                "text": event.get("content", event.get("text", event.get("summary", ""))),
                                "summary": event.get("summary", event.get("content", "")),
                                "timestamp_utc": event.get("timestamp", event.get("updated_at", "")),
                            })
                    
                    # Also check for related nodes/edges if present
                    related_nodes = graph_response.get("related_nodes", [])
                    for node in related_nodes:
                        if isinstance(node, dict) and node.get("type") == "Event":
                            results.append({
                                "id": node.get("uuid", ""),
                                "source": "knowledge_graph",
                                "text": node.get("content", node.get("name", "")),
                                "timestamp_utc": node.get("updated_at", ""),
                            })
                            
                elif isinstance(graph_response, str):
                    results.append({
                        "id": "kg_response",
                        "source": "knowledge_graph",
                        "text": graph_response,
                    })
                elif isinstance(graph_response, list):
                    for item in graph_response:
                        if isinstance(item, dict):
                            results.append({
                                "id": item.get("id", item.get("uuid", "")),
                                "source": "knowledge_graph",
                                "person_id": item.get("person_id", ""),
                                "person_name": item.get("person_name", item.get("name", "")),
                                "text": item.get("text", item.get("content", "")),
                                "timestamp_utc": item.get("timestamp", ""),
                            })
                    
        except Exception as e:
            # Handle the "No center node provided" error gracefully
            if "No center node provided" in str(e):
                print(f"[DEBUG] Knowledge graph query requires person_id, skipping global search")
            else:
                print(f"[DEBUG] Knowledge graph query failed: {e}")
        
        return results
    
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
                # Now using await since retrieve_memory is async
                result = await self.retrieve_memory(args.get("queryText", ""))
                
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
        
        # Close MemobotService connection
        if self.memobot_service:
            try:
                asyncio.run(self.memobot_service.close())
            except:
                pass


def run_realtime_mode(user_name=None, person_id=None):
    """Run the agent in realtime audio mode.

    Args:
        user_name: Optional name of the recognized user; injected into the system prompt.
        person_id: Optional person_id for knowledge graph queries.
    """
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
    if user_name:
        print(f"Recognized user: {user_name}")
    if person_id:
        print(f"Person ID: {person_id}")
    print("This mode uses OpenAI's Realtime API for")
    print("voice-to-voice conversation with memory access.")
    print("\nRequirements:")
    print("  - sounddevice (pip install sounddevice)")
    print("  - numpy (pip install numpy)")
    print("  - websockets (pip install websockets)")
    print("=" * 40)
    
    agent = RealtimeAgent(api_key, user_name=user_name, person_id=person_id)
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


async def test_knowledge_graph():
    """Test the knowledge graph service connection and basic operations."""
    print("\n" + "=" * 60)
    print("🧪 Testing Knowledge Graph Service")
    print("=" * 60)
    
    # Check if MemobotService is available
    if MemobotService is None:
        print("\n❌ FAILED: Memobot package not found")
        print("   Make sure the Memobot package exists in the root directory")
        print(f"   Expected location: {_REPO_ROOT}/Memobot/")
        print("\n   Directory contents:")
        try:
            for item in os.listdir(_REPO_ROOT):
                print(f"      {item}")
        except Exception as e:
            print(f"      Error listing directory: {e}")
        return False
    
    print("\n✅ Memobot package imported successfully")
    
    # Initialize the service
    try:
        print("\n📡 Initializing MemobotService...")
        service = MemobotService.from_env(group_id='tenant_001')
        print("✅ MemobotService initialized successfully")
    except Exception as e:
        print(f"\n❌ FAILED to initialize MemobotService: {e}")
        print("\n   Check your environment variables in .env:")
        print("   - NEO4J_URI")
        print("   - NEO4J_USER")
        print("   - NEO4J_PASSWORD")
        return False
    
    # Test 1: Check service attributes
    print("\n📋 Service attributes:")
    print(f"   - group_id: {getattr(service, 'group_id', 'N/A')}")
    print(f"   - Service type: {type(service).__name__}")
    print(f"   - Available methods: {[m for m in dir(service) if not m.startswith('_')]}")
    
    # Test 2: Test build operation with sample data
    print("\n🏗️  Testing build operation...")
    from datetime import datetime, timezone
    
    test_data = {
        "id": f"test_log_{int(datetime.now().timestamp())}",
        "person_name": "Test User",
        "person_id": "test_person_001",
        "text": "This is a test memory entry for knowledge graph verification.",
        "robot_pos_list": [],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    
    try:
        await service.build(test_data)
        print("   ✅ Build operation successful")
    except Exception as e:
        print(f"   ⚠️  Build failed: {e}")
    
    # Test 3: Test retrieve operation
    test_queries = [
        ("What is the test memory?", "test_person_001"),
        ("What happened?", None),
    ]
    
    print("\n🔍 Testing retrieve queries...")
    for query, person_id in test_queries:
        print(f"\n   Query: '{query}' (person_id={person_id})")
        try:
            result = await service.retrieve(query, person_id=person_id)
            print(f"   ✅ Retrieve successful")
            print(f"   📄 Result type: {type(result).__name__}")
            print(f"   📄 Result: {str(result)[:200]}...")
        except Exception as e:
            print(f"   ⚠️  Retrieve failed: {e}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    try:
        await service.close()
        print("   ✅ Service closed successfully")
    except Exception as e:
        print(f"   ⚠️  Close failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Knowledge Graph Service test completed")
    print("=" * 60)
    return True


def run_test_graph():
    """Run the knowledge graph test."""
    success = asyncio.run(test_knowledge_graph())
    sys.exit(0 if success else 1)


async def add_test_memory():
    """Add a test memory entry to the knowledge graph."""
    print("\n" + "=" * 60)
    print("📝 Adding Test Memory to Knowledge Graph")
    print("=" * 60)
    
    if MemobotService is None:
        print("\n❌ FAILED: Memobot package not found")
        return False
    
    try:
        print("\n📡 Initializing MemobotService...")
        service = MemobotService.from_env(group_id='tenant_001')
        print("✅ MemobotService initialized successfully")
    except Exception as e:
        print(f"\n❌ FAILED to initialize MemobotService: {e}")
        return False
    
    # Add Jason wearing black jacket - following the example format
    input_data = {
        "id": "log_jason_001",
        "person_name": "Jason",
        "person_id": "user_jason_001",
        "text": "Jason is wearing a black jacket today. He looks very stylish.",
        "robot_pos_list": [],
        "timestamp": "2026-01-28T10:00:00Z",
    }
    
    print(f"\n📄 Adding memory:")
    print(f"   ID: {input_data['id']}")
    print(f"   Person: {input_data['person_name']}")
    print(f"   Person ID: {input_data['person_id']}")
    print(f"   Text: {input_data['text']}")
    print(f"   Timestamp: {input_data['timestamp']}")
    
    try:
        await service.build(input_data)
        print("\n✅ Memory added successfully!")
    except Exception as e:
        print(f"\n❌ Failed to add memory: {e}")
        await service.close()
        return False
    
    # Verify by querying
    print("\n🔍 Verifying by querying...")
    try:
        result = await service.retrieve(
            "What is Jason wearing?",
            person_id="user_jason_001",
        )
        print(f"✅ Query successful")
        print(f"📄 Result: {result}")
    except Exception as e:
        print(f"⚠️  Query failed: {e}")
    
    # Cleanup
    await service.close()
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)
    return True


def run_add_memory():
    """Run the add memory function."""
    success = asyncio.run(add_test_memory())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memobot - AI assistant with memory")
    parser.add_argument(
        "--mode", 
        choices=["text", "realtime", "audio", "test-graph", "add-memory"],
        default="text",
        help="Interaction mode: 'text' for chat, 'realtime' or 'audio' for voice, 'test-graph' to test knowledge graph, 'add-memory' to add test data"
    )
    parser.add_argument(
        "--user-name",
        default=None,
        help="Recognized user's name; injected into the system prompt in realtime/audio mode"
    )
    parser.add_argument(
        "--person-id",
        default=None,
        help="Person ID for knowledge graph queries (e.g., user_jason_001)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test-graph":
        run_test_graph()
    elif args.mode == "add-memory":
        run_add_memory()
    elif args.mode in ["realtime", "audio"]:
        run_realtime_mode(user_name=args.user_name, person_id=args.person_id)
    else:
        run_text_mode()
