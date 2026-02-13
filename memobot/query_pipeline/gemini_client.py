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
    from google.genai.types import (
        Content,
        FunctionDeclaration,
        LiveConnectConfig,
        Part,
        PrebuiltVoiceConfig,
        SpeechConfig,
        Tool,
        VoiceConfig,
    )
    from google.genai.types import Schema
    from google.genai.types import Type as SchemaType
    print("[Info] google-genai package loaded successfully")
except ImportError:
    genai = None
    types = None
    Content = Part = LiveConnectConfig = SpeechConfig = VoiceConfig = None
    PrebuiltVoiceConfig = Tool = FunctionDeclaration = Schema = SchemaType = None
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
VOICE_NAME = "Aoede"

BASE_INSTRUCTIONS = (
    "You are a helpful voice assistant. Speak English only. Keep replies short.\n\n"
    "MEMORY: If the user asks about the past, preferences, people, or anything that might be in memory, call retrieveMemory first. "
    "Do not answer from memory without calling it. Do not guess. If unsure, say so.\n\n"
    "RULES: Do not make up facts. Be concise."
)


def _get_retrieve_memory_tool():
    """Return a Tool for retrieveMemory (typed for Live API)."""
    if Tool is None or FunctionDeclaration is None or Schema is None or SchemaType is None:
        return None
    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name="retrieveMemory",
                description="Retrieve past memories based on a natural language query.",
                parameters=Schema(
                    type=SchemaType.OBJECT,
                    properties={
                        "queryText": Schema(
                            type=SchemaType.STRING,
                            description="The natural language query to search for memories.",
                        ),
                    },
                    required=["queryText"],
                ),
            )
        ]
    )


def _get_extra_tools():
    """Return a Tool with searchRobotActions, writeCode, executeCode, saveCode (typed for Live API)."""
    if Tool is None or FunctionDeclaration is None or Schema is None or SchemaType is None:
        return None
    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name="searchRobotActions",
                description="Search for available robot actions (e.g. move, grasp, speak) by keyword or category.",
                parameters=Schema(
                    type=SchemaType.OBJECT,
                    properties={
                        "query": Schema(
                            type=SchemaType.STRING,
                            description="Search query for robot actions.",
                        ),
                        "category": Schema(
                            type=SchemaType.STRING,
                            description="Optional category to filter actions (e.g. movement, manipulation, speech).",
                        ),
                    },
                    required=["query"],
                ),
            ),
            FunctionDeclaration(
                name="writeCode",
                description="Write or generate code in a given language. Returns the generated code.",
                parameters=Schema(
                    type=SchemaType.OBJECT,
                    properties={
                        "prompt": Schema(
                            type=SchemaType.STRING,
                            description="Description of what the code should do.",
                        ),
                        "language": Schema(
                            type=SchemaType.STRING,
                            description="Programming language (e.g. python, javascript).",
                        ),
                    },
                    required=["prompt"],
                ),
            ),
            FunctionDeclaration(
                name="executeCode",
                description="Execute code in a sandboxed environment and return stdout, stderr, and result.",
                parameters=Schema(
                    type=SchemaType.OBJECT,
                    properties={
                        "code": Schema(
                            type=SchemaType.STRING,
                            description="The code to execute.",
                        ),
                        "language": Schema(
                            type=SchemaType.STRING,
                            description="Programming language of the code (e.g. python, javascript).",
                        ),
                    },
                    required=["code"],
                ),
            ),
            FunctionDeclaration(
                name="saveCode",
                description="Save code to a file or named snippet in the project.",
                parameters=Schema(
                    type=SchemaType.OBJECT,
                    properties={
                        "code": Schema(
                            type=SchemaType.STRING,
                            description="The code to save.",
                        ),
                        "filename": Schema(
                            type=SchemaType.STRING,
                            description="Filename or path to save the code to.",
                        ),
                    },
                    required=["code", "filename"],
                ),
            ),
        ]
    )


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

    def interrupt(self, print_message=False):
        """Handle interruption by clearing queue. Set print_message=True only when user interrupted (not when AI finished)."""
        try:
            while not self.play_queue.empty():
                self.play_queue.get_nowait()
                self.play_queue.task_done()
        except queue.Empty:
            pass
        
        # Reset last playback time to allow immediate mic input
        with self._lock:
            self.last_playback_time = 0
        if print_message:
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
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1alpha"},
        )
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
        if LiveConnectConfig is None:
            # Fallback dict config if google-genai types not available
            return {
                "response_modalities": ["AUDIO"],
                "system_instruction": instructions,
                "input_audio_transcription": {},
                "output_audio_transcription": {},
                "tools": [
                    {
                        "function_declarations": [{
                            "name": "retrieveMemory",
                            "description": "Retrieve past memories based on a natural language query.",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {"queryText": {"type": "STRING", "description": "Natural language query."}},
                                "required": ["queryText"],
                            },
                        }]
                    },
                    {
                        "function_declarations": [
                            {"name": "searchRobotActions", "description": "Search for available robot actions.", "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}, "category": {"type": "STRING"}}, "required": ["query"]}},
                            {"name": "writeCode", "description": "Write or generate code.", "parameters": {"type": "OBJECT", "properties": {"prompt": {"type": "STRING"}, "language": {"type": "STRING"}}, "required": ["prompt"]}},
                            {"name": "executeCode", "description": "Execute code in a sandbox.", "parameters": {"type": "OBJECT", "properties": {"code": {"type": "STRING"}, "language": {"type": "STRING"}}, "required": ["code"]}},
                            {"name": "saveCode", "description": "Save code to a file.", "parameters": {"type": "OBJECT", "properties": {"code": {"type": "STRING"}, "filename": {"type": "STRING"}}, "required": ["code", "filename"]}},
                        ]
                    },
                ],
            }
        memory_tool = _get_retrieve_memory_tool()
        extra_tool = _get_extra_tools()
        tools = [t for t in [memory_tool, extra_tool] if t is not None]
        return LiveConnectConfig(
            output_audio_transcription={},
            input_audio_transcription={},
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=VOICE_NAME)
                )
            ),
            system_instruction=Content(parts=[Part(text=instructions)]),
            tools=tools,
        )

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
                func_response = types.FunctionResponse(
                    id=call_id,
                    name=name,
                    response={"result": result},
                )
                function_responses.append(func_response)
            elif name == "searchRobotActions":
                query = args.get("query", "")
                category = args.get("category", "")
                # Placeholder result
                result = {
                    "actions": [
                        {"id": "move_forward", "name": "Move Forward", "category": "movement", "description": "[placeholder] Move robot forward."},
                        {"id": "grasp", "name": "Grasp", "category": "manipulation", "description": "[placeholder] Grasp object."},
                        {"id": "speak", "name": "Speak", "category": "speech", "description": "[placeholder] Speak text."},
                    ],
                    "query": query,
                    "category": category or None,
                }
                function_responses.append(types.FunctionResponse(id=call_id, name=name, response={"result": result}))
            elif name == "writeCode":
                prompt = args.get("prompt", "")
                language = args.get("language", "python")
                # Placeholder result
                result = {
                    "code": "# [placeholder] Generated code based on prompt\n# prompt: " + prompt[:100] + ("..." if len(prompt) > 100 else "") + "\ndef placeholder():\n    pass",
                    "language": language,
                    "prompt": prompt,
                }
                function_responses.append(types.FunctionResponse(id=call_id, name=name, response={"result": result}))
            elif name == "executeCode":
                code = args.get("code", "")
                language = args.get("language", "python")
                # Placeholder result
                result = {
                    "stdout": "[placeholder] Execution stdout",
                    "stderr": "",
                    "result": "[placeholder] Return value",
                    "exit_code": 0,
                    "language": language,
                }
                function_responses.append(types.FunctionResponse(id=call_id, name=name, response={"result": result}))
            elif name == "saveCode":
                code = args.get("code", "")
                filename = args.get("filename", "")
                # Placeholder result
                result = {
                    "saved": True,
                    "filename": filename,
                    "message": "[placeholder] Code saved successfully",
                    "bytes_written": len(code.encode("utf-8")),
                }
                function_responses.append(types.FunctionResponse(id=call_id, name=name, response={"result": result}))
        
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
        """Receive responses from Gemini (matches working Live API receive pattern)."""
        try:
            while True:
                turn = self.session.receive()
                ai_text_started = False  # Print "Gemini said:" prefix once per turn
                async for response in turn:
                    # Audio: use response.data (aggregated inline_data) when present
                    if response.data:
                        self.audio_mgr.queue_audio(response.data)
                    # Text: print what the AI says (with prefix once per turn)
                    if response.text:
                        if not ai_text_started:
                            print("\n🗣️ Gemini said: ", end="")
                            ai_text_started = True
                        print(response.text, end="")

                    server_content = response.server_content
                    if server_content:
                        # Print user speech (transcription; API returns object with .text when input_audio_transcription enabled)
                        if getattr(server_content, "input_transcription", None):
                            it = server_content.input_transcription
                            text = it.get("text", it) if isinstance(it, dict) else getattr(it, "text", it)
                            if text:
                                print(f"\n👤 You said: {text}")
                        # Print AI speech transcript when available
                        if getattr(server_content, "output_transcription", None):
                            ot = server_content.output_transcription
                            text = ot.get("text", ot) if isinstance(ot, dict) else getattr(ot, "text", ot)
                            if text:
                                if not ai_text_started:
                                    print("\n🗣️ Gemini said: ", end="")
                                    ai_text_started = True
                                print(text, end="")
                        # Only print "Audio playback interrupted" when user talked over AI, not when AI finished
                        if server_content.interrupted:
                            self.audio_mgr.interrupt(print_message=True)
                        elif server_content.turn_complete:
                            if ai_text_started:
                                print()  # Newline after AI response
                            self.audio_mgr.interrupt(print_message=False)

                    if response.tool_call:
                        print("\n[DEBUG] Tool call received")
                        await self.handle_tool_call(response.tool_call)

                # After turn ends: clear queue so playback stops (no message—AI just finished)
                self.audio_mgr.interrupt(print_message=False)
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
        _patch_live_setup_for_transcription()
        _patch_live_parse_transcription()

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


def _patch_live_setup_for_transcription():
    """Patch SDK so setup message includes input/output_audio_transcription (API requires it for transcriptions)."""
    try:
        from google.genai import live as live_module
        _orig_mldev = live_module.AsyncLive._LiveSetup_to_mldev

        def _patched_mldev(self, model, config=None):
            result = _orig_mldev(self, model, config)
            from_object = config.model_dump(exclude_none=True) if isinstance(config, types.LiveConnectConfig) else (config or {})
            if "setup" not in result:
                return result
            # Always request transcriptions so API returns "You said" / "Gemini said" (SDK may drop these from config)
            result["setup"]["inputAudioTranscription"] = from_object.get("input_audio_transcription", {})
            result["setup"]["outputAudioTranscription"] = from_object.get("output_audio_transcription", {})
            return result

        live_module.AsyncLive._LiveSetup_to_mldev = _patched_mldev
    except Exception:
        pass


def _patch_live_parse_transcription():
    """Patch SDK to pass through inputTranscription/outputTranscription from API into server_content."""
    try:
        from typing import Any, Optional
        from google.genai import live as live_module
        from google.genai import types as genai_types
        _orig_content_from_mldev = live_module.AsyncLive._LiveServerContent_from_mldev

        def _patched_content_from_mldev(self, from_object):
            to_object = _orig_content_from_mldev(self, from_object)
            sc = from_object if isinstance(from_object, dict) else getattr(from_object, "__dict__", {})
            if sc.get("inputTranscription") is not None:
                setattr(to_object, "input_transcription", sc["inputTranscription"])
            if sc.get("outputTranscription") is not None:
                setattr(to_object, "output_transcription", sc["outputTranscription"])
            return to_object

        live_module.AsyncLive._LiveServerContent_from_mldev = _patched_content_from_mldev

        # Subclass so parsed messages have .input_transcription / .output_transcription (SDK model ignores extra)
        _base = genai_types.LiveServerContent
        if not hasattr(_base, "input_transcription"):
            class _LiveServerContentWithTranscription(_base):
                input_transcription: Optional[Any] = None
                output_transcription: Optional[Any] = None
            genai_types.LiveServerContent = _LiveServerContentWithTranscription
    except Exception:
        pass


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
    parser.add_argument("--user-name", default=None)
    parser.add_argument("--person-id", default=None)
    args = parser.parse_args()
    run_realtime_mode(user_name=args.user_name, person_id=args.person_id)