import json
import os
import sys
import asyncio
import threading
import queue
import time
from collections import deque
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    print("[Info] google-genai package loaded successfully")
except ImportError:
    genai = None
    types = None
    print("[Warning] google-genai package not found. Install with: pip install google-genai")

try:
    import pyaudio
    print("[Info] pyaudio loaded successfully")
except ImportError:
    pyaudio = None
    print("[Warning] pyaudio not found. Install with: pip install pyaudio")

try:
    from .agent import Agent
except ImportError:
    from agent import Agent

# Load env from repo root (two levels up from query_pipeline)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOTENV_PATH = os.path.join(_REPO_ROOT, ".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

MemobotService = None
DEFAULT_MEMOBOT_GROUP_ID = os.getenv("MEMOBOT_GROUP_ID", "tenant_001")

try:
    from Memobot import MemobotService
    print("[Info] Memobot package loaded successfully")
except ImportError:
    pass

# Audio config
FORMAT = pyaudio.paInt16 if pyaudio else None
CHANNELS = 1
SEND_SAMPLE_RATE = 16000      # Gemini expects 16kHz input
RECEIVE_SAMPLE_RATE = 24000   # Gemini outputs 24kHz audio
CHUNK_SIZE = 512              # Small chunks for lower latency

# Echo cancellation settings
PLAYBACK_GRACE_PERIOD = 0.5   # Increased to 500ms for better echo prevention
MIC_QUEUE_MAXSIZE = 3         # Reduced to prevent audio backlog

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

BASE_INSTRUCTIONS = (
    "You are a helpful voice assistant. Speak English only. Keep replies short.\n\n"
    "MEMORY: If the user asks about the past, preferences, people, or anything that might be in memory, call retrieveMemory first. "
    "Do not answer from memory without calling it. Do not guess. If unsure, say so.\n\n"
    "RULES: Do not make up facts. Be concise."
)


def _get_retrieve_memory_tool():
    return {
        "function_declarations": [{
            "name": "retrieveMemory",
            "description": "Retrieve past memories based on a natural language query.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "queryText": {
                        "type": "STRING",
                        "description": "The natural language query to search for memories.",
                    },
                },
                "required": ["queryText"],
            },
        }]
    }


class AudioManager:
    def __init__(self):
        self.p = pyaudio.PyAudio() if pyaudio else None
        self.in_stream = None
        self.out_stream = None
        self.play_queue = queue.Queue(maxsize=100)  # Bounded queue to prevent excessive buffering
        self.playback_thread = None
        self.is_running = True
        self.last_playback_time = 0  # Timestamp when last audio finished playing
        self._lock = threading.Lock()

    async def init(self):
        if not self.p:
            return
        
        try:
            mic_info = self.p.get_default_input_device_info()
            # Mic input - optimized for minimal latency
            self.in_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=None,  # Use blocking mode for simplicity
            )
            # Speaker output - optimized for minimal latency
            self.out_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE * 2,  # Slightly larger output buffer
            )
            print("🎤 Mic + 🔈 Speaker initialized")
            print(f"📊 Audio Config: Input={SEND_SAMPLE_RATE}Hz, Output={RECEIVE_SAMPLE_RATE}Hz, Chunk={CHUNK_SIZE}")
            
            # Start dedicated playback thread
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            raise

    def read_mic(self):
        # Check if we're in grace period after playback
        with self._lock:
            time_since_playback = time.time() - self.last_playback_time
            if time_since_playback < PLAYBACK_GRACE_PERIOD:
                return b'\x00' * (CHUNK_SIZE * 2)  # Return silence during grace period
            
        try:
            if self.in_stream:
                return self.in_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            return b'\x00' * (CHUNK_SIZE * 2)
        except Exception as e:
            print(f"Error reading microphone: {e}")
            return b'\x00' * (CHUNK_SIZE * 2)

    def queue_audio(self, data: bytes):
        if self.is_running:
            self.play_queue.put(data)

    def _playback_loop(self):
        while self.is_running:
            try:
                if self.play_queue.empty():
                    time.sleep(0.001)  # Very short sleep when idle
                    continue
                    
                data = self.play_queue.get(timeout=0.05)
                if self.out_stream:
                    self.out_stream.write(data)
                
                # Update last playback time
                with self._lock:
                    self.last_playback_time = time.time()
                    
                self.play_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error playing audio: {e}")
                time.sleep(0.05)

    def interrupt(self):
        """Handle interruption by clearing queue"""
        try:
            while not self.play_queue.empty():
                self.play_queue.get_nowait()
                self.play_queue.task_done()
        except queue.Empty:
            pass
        
        # Reset last playback time to allow immediate mic input
        with self._lock:
            self.last_playback_time = 0
        print("🔇 Audio playback interrupted")

    def cleanup(self):
        self.is_running = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
            
        if self.in_stream:
            try:
                self.in_stream.stop_stream()
                self.in_stream.close()
            except: pass
        if self.out_stream:
            try:
                self.out_stream.stop_stream()
                self.out_stream.close()
            except: pass
        if self.p:
            self.p.terminate()
        print("🎤 Mic + 🔈 Speaker cleaned up")


class RealtimeAgent:
    def __init__(self, api_key=None, user_name=None, person_id=None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.user_name = user_name
        self.person_id = person_id
        self.client = genai.Client(api_key=self.api_key)
        self.session = None
        self.memobot_service = None
        self.audio_mgr = AudioManager()
        
        # Queue for sending mic audio to gemini - smaller to reduce latency
        self.audio_queue_mic = asyncio.Queue(maxsize=MIC_QUEUE_MAXSIZE)
        
        self._init_memobot_service()
    
    def _init_memobot_service(self):
        if MemobotService is None:
            return
        try:
            self.memobot_service = MemobotService.from_env(group_id=DEFAULT_MEMOBOT_GROUP_ID)
            print("[Info] MemobotService initialized")
        except Exception as e:
            print(f"[Warning] Failed to initialize MemobotService: {e}")

    def _get_config(self):
        instructions = BASE_INSTRUCTIONS
        if self.user_name:
            instructions += f"\n\nThe user is {self.user_name}."
        
        return {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Puck"
                    }
                }
            },
            "system_instruction": instructions,
            "tools": [_get_retrieve_memory_tool()],
        }

    async def retrieve_memory(self, query):
        """Retrieve memories from vector DB and knowledge graph."""
        print(f"\n🔍 [Retrieving] Query: '{query}'")
        results = []
        
        # Vector DB
        try:
            try:
                from .query import retrieve_and_rank
            except ImportError:
                from query import retrieve_and_rank
            
            print(f"[DEBUG] Querying vector DB: {query}")
            db_results = retrieve_and_rank(
                question=query, index_name="memobot-memories", top_k=5,
                alpha=0.5, beta=0.3, gamma=0.2, person_id=self.person_id,
            )
            for r in db_results:
                md = r.get("metadata", {})
                results.append({"source": "vector_db", "summary": md.get("summary", "")})
        except Exception as e:
            print(f"[DEBUG] Vector DB error: {e}")
        
        # Knowledge Graph
        if False and self.memobot_service and self.person_id: # Temporarily disabled
            try:
                print(f"[DEBUG] Querying KG with person_id: {self.person_id}")
                graph_response = await self.memobot_service.retrieve(query, person_id=self.person_id)
                if graph_response and isinstance(graph_response, dict):
                    for event in graph_response.get("events", []):
                        if isinstance(event, dict):
                            results.append({
                                "source": "knowledge_graph",
                                "summary": event.get("content", event.get("summary", "")),
                            })
            except Exception as e:
                print(f"[DEBUG] KG error: {e}")
        
        print(f"📋 Retrieved {len(results)} memories")
        for i, r in enumerate(results[:3], 1):
            print(f"  [{i}] Source: {r.get('source')} | {r.get('summary', 'N/A')[:500]}...")
        
        return {"events": results, "query": query}

    async def handle_tool_call(self, tool_call):
        """Handle function calls from Gemini."""
        function_calls = getattr(tool_call, 'function_calls', []) or []
        function_responses = []
        
        for fc in function_calls:
            name = getattr(fc, 'name', None)
            call_id = getattr(fc, 'id', None)
            args = getattr(fc, 'args', {})
            
            if hasattr(args, 'items'):
                args = dict(args)
            
            print(f"[DEBUG] Function call: {name}, id={call_id}, args={args}")
            
            if name == "retrieveMemory":
                query_text = args.get('queryText', '')
                result = await self.retrieve_memory(query_text)
                
                # Create function response
                func_response = types.FunctionResponse(
                    id=call_id,
                    name=name,
                    response={"result": result}, # Pass dict directly, no need to json.dump again usually
                )
                function_responses.append(func_response)
        
        # Send all function responses at once
        if function_responses:
            try:
                # Wrap responses in LiveClientToolResponse and use generic send()
                tool_response = types.LiveClientToolResponse(function_responses=function_responses)
                await self.session.send(input=tool_response)
                print(f"[DEBUG] Tool response sent successfully")
            except Exception as e:
                print(f"[DEBUG] Tool response error: {e}")

    async def listen_audio(self):
        """Capture audio from microphone and queue it."""
        try:
            while self.audio_mgr.is_running:
                data = await asyncio.to_thread(self.audio_mgr.read_mic)
                try:
                    self.audio_queue_mic.put_nowait({"data": data, "mime_type": "audio/pcm"})
                except asyncio.QueueFull:
                    # Drop oldest frame if queue is full to maintain real-time behavior
                    try:
                        self.audio_queue_mic.get_nowait()
                        self.audio_queue_mic.put_nowait({"data": data, "mime_type": "audio/pcm"})
                    except:
                        pass
                # Sleep for chunk duration to maintain timing
                await asyncio.sleep(CHUNK_SIZE / SEND_SAMPLE_RATE)
        except asyncio.CancelledError:
            print("[DEBUG] listen_audio cancelled")
            raise

    async def send_audio(self):
        """Send queued audio to Gemini."""
        try:
            while True:
                msg = await self.audio_queue_mic.get()
                # Use send() with the audio data directly
                await self.session.send(input=msg)
        except asyncio.CancelledError:
            print("[DEBUG] send_audio cancelled")
            raise

    async def receive_responses(self):
        """Receive responses from Gemini."""
        try:
            while True:
                async for response in self.session.receive():
                    server_content = response.server_content
                    # Handle server content if present
                    if server_content:
                        # Check for user input transcription (if provided by the API)
                        if hasattr(server_content, 'model_turn'): 
                            pass # placeholder for existing logic structure

                        # Attempt to print user transcript if available
                        if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                            print(f"\n🗣️  [You] {server_content.input_transcription}")

                        # Check for user input transcription (User speech turned to text)
                        model_turn = server_content.model_turn
                        turn_complete = server_content.turn_complete
                        
                        # Sometimes transcription comes as a specialized field
                        # Let's inspect the object structure more broadly if needed, but SDK usually exposes it here.
                        # Note: The property name might depend on specific API version (e.g., streaming recognition result).
                        # We will try to catch model output text specifically.

                        if model_turn:
                            for part in model_turn.parts:
                                # Text output (Model thoughts/text response)
                                if part.text:
                                    print(f"🤖 [Gemini Text] {part.text}")
                                
                                # Audio output
                                if part.inline_data and isinstance(part.inline_data.data, bytes):
                                    self.audio_mgr.queue_audio(part.inline_data.data)

                        # Check for turn complete
                        if turn_complete:
                            # print()  # New line after turn completes
                            pass
                        
                        # Check for interrupted - clear audio queue
                        if server_content.interrupted:
                            print("\n[Interrupted]")
                            # Clear the output queue on interruption
                            self.audio_mgr.interrupt()
                    
                    # Handle independent audio data if any (legacy or specific stream types)
                    if response.data:
                         self.audio_mgr.queue_audio(response.data)

                    # Handle tool calls
                    if response.tool_call:
                        print("\n[DEBUG] Tool call received")
                        await self.handle_tool_call(response.tool_call)

                    # Inspect raw server content for transcriptions (if available in this SDK version)
                    # Some versions put "speech_recognition_results" or similar outside server_content
                    # For now, we rely on standard logging.
                    
        except asyncio.CancelledError:
            print("[DEBUG] receive_responses cancelled")
            raise
        except Exception as e:
            print(f"[ERROR] receive_responses error: {e}")
            import traceback
            traceback.print_exc()

    async def run(self):
        """Main run loop."""
        print("🎙️  Initializing Audio Manager...")
        await self.audio_mgr.init()
        
        config = self._get_config()
        
        try:
            async with self.client.aio.live.connect(model=MODEL, config=config) as session:
                self.session = session
                print("✅ Connected to Gemini Live API!")
                print("🎙️  Start speaking...\n")
                
                # Create tasks manually for better control
                tasks = [
                    asyncio.create_task(self.listen_audio(), name="listen"),
                    asyncio.create_task(self.send_audio(), name="send"),
                    asyncio.create_task(self.receive_responses(), name="receive"),
                ]
                
                try:
                    # Wait for any task to complete (which would indicate an error)
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    
                    # Check if any task raised an exception
                    for task in done:
                        if task.exception():
                            print(f"[ERROR] Task {task.get_name()} failed: {task.exception()}")
                finally:
                    # Cancel all remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Wait for all tasks to finish cancellation
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ERROR] Run error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        self.audio_mgr.cleanup()
        if self.memobot_service:
            try:
                await self.memobot_service.close()
            except:
                pass
        print("\n👋 Connection closed.")


def run_realtime_mode(user_name=None, person_id=None):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\nERROR: GOOGLE_API_KEY is not set.")
        sys.exit(1)
    
    if not pyaudio:
        print("\nERROR: pyaudio is required. Install with: pip install pyaudio")
        sys.exit(1)
    
    print("\n🎙️  Memobot Realtime Audio Mode (Gemini)")
    print("=" * 40)
    if user_name:
        print(f"User: {user_name}")
    if person_id:
        print(f"Person ID: {person_id}")
    print("=" * 40)
    
    agent = RealtimeAgent(api_key, user_name=user_name, person_id=person_id)
    
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memobot - AI assistant with memory")
    parser.add_argument("--mode", choices=["text", "realtime", "audio"], default="realtime")
    parser.add_argument("--user-name", default=None)
    parser.add_argument("--person-id", default=None)
    args = parser.parse_args()
    
    if args.mode in ["realtime", "audio"]:
        run_realtime_mode(user_name=args.user_name, person_id=args.person_id)
    else:
        print("Text mode not implemented in this version")