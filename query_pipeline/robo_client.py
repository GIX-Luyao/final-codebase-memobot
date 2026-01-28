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

# Audio constants for Realtime API (24kHz, 16-bit PCM, mono)
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_DURATION_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class RealtimeAgent:
    """Agent using OpenAI's Realtime API for audio-to-audio interaction."""
    
    def __init__(self, api_key, user_name=None):
        self.api_key = api_key
        self.user_name = user_name
        self.ws = None
        self.audio_queue = asyncio.Queue()
        self.is_recording = False
        self.input_stream = None
        self.output_stream = None
        self.sd = None  # sounddevice module
        
    async def get_ephemeral_token(self):
        """Get an ephemeral client secret for WebSocket connection."""
        base_instructions = "You are a helpful assistant with access to a memory system. When users ask about past events, use the retrieveMemory function to find relevant information. When you are unsure, do not make up information. Keep responses concise for voice interaction."
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
    
    def retrieve_memory(self, query):
        """Retrieve memories using the existing pipeline."""
        try:
            # Direct import since we're in the same directory
            from query import retrieve_and_rank
            
            print(f"[DEBUG] Calling retrieve_and_rank with query: {query}")
            
            results = retrieve_and_rank(
                question=query,
                index_name="twelve-labs",
                top_k=10,
                alpha=0.5,
                beta=0.3,
                gamma=0.2,
            )
            
            events = []
            for r in results:
                md = r.get("metadata", {})
                event = {
                    "id": r["id"],
                    "summary": md.get("summary", ""),
                    "video_file": md.get("video_file", ""),
                    "start_time_sec": md.get("start_time_sec"),
                    "end_time_sec": md.get("end_time_sec"),
                    "timestamp_utc": md.get("timestamp_utc"),
                    "importance_score": md.get("importance_score"),
                    "relevance_score": r["relevance_score"],
                    "final_score": r["final_score"],
                }
                events.append(event)
            
            memory_context = {
                "events": events,
                "metadata": {
                    "total_results": len(events),
                    "query": query,
                }
            }
            
            if memory_context.get("events"):
                print(f"\n📋 Retrieved {len(memory_context['events'])} relevant memories:")
                for i, event in enumerate(events[:5], 1):  # Show top 5
                    print(f"\n  [{i}] ID: {event['id']}")
                    print(f"      Summary: {event['summary'][:150]}..." if len(event.get('summary', '')) > 150 else f"      Summary: {event.get('summary', 'N/A')}")
                    print(f"      Video: {event.get('video_file', 'N/A')}")
                    print(f"      Time: {event.get('start_time_sec', '?')}s - {event.get('end_time_sec', '?')}s")
                    print(f"      Timestamp: {event.get('timestamp_utc', 'N/A')}")
                    print(f"      Scores: relevance={event.get('relevance_score', 0):.3f}, final={event.get('final_score', 0):.3f}")
                if len(events) > 5:
                    print(f"\n  ... and {len(events) - 5} more memories")
                print()
            else:
                print(f"[DEBUG] No events found")
            
            return memory_context
            
        except ImportError as e:
            print(f"[DEBUG] Import error: {e}")
            print("[DEBUG] Make sure you're running with the correct virtual environment:")
            print("  source venv/bin/activate  # or your venv path")
            print("  python query_pipeline/robo_client.py --mode realtime")
            return {"events": [], "metadata": {"error": f"Import error: {str(e)}"}}
        except Exception as e:
            import traceback
            print(f"[DEBUG] Exception in retrieve_memory: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
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


def run_realtime_mode(user_name=None):
    """Run the agent in realtime audio mode.

    Args:
        user_name: Optional name of the recognized user; injected into the system prompt.
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
    print("This mode uses OpenAI's Realtime API for")
    print("voice-to-voice conversation with memory access.")
    print("\nRequirements:")
    print("  - sounddevice (pip install sounddevice)")
    print("  - numpy (pip install numpy)")
    print("  - websockets (pip install websockets)")
    print("=" * 40)
    
    agent = RealtimeAgent(api_key, user_name=user_name)
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
    parser.add_argument(
        "--user-name",
        default=None,
        help="Recognized user's name; injected into the system prompt in realtime/audio mode"
    )
    
    args = parser.parse_args()
    
    if args.mode in ["realtime", "audio"]:
        run_realtime_mode(user_name=args.user_name)
    else:
        run_text_mode()
