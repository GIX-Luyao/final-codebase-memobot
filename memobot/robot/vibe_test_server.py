#!/usr/bin/env python3
import socket
import struct
import threading
import os
import re
import io
import time
import wave
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types  # Required for sending audio bytes

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT_AUDIO_RX    = 50005  # Robot Mic -> Server
PORT_CMD_TX      = 50007  # Server Code -> Robot
PORT_FEEDBACK_RX = 50008  # Robot Logs -> Server

# Gemini Settings
# Use 'gemini-2.0-flash' or 'gemini-1.5-pro' (Change if needed)
MODEL_ID = "gemini-3-pro-preview" 

# VAD Settings
VAD_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
SILENCE_DURATION_LIMIT = 1.0 

# --- SETUP CLIENT ---
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in environment.")
    exit()

print(f"🔑 Using API Key: {api_key[:5]}...{api_key[-3:]}")

try:
    client = genai.Client(api_key=api_key)
    print(f"🚀 Connecting to {MODEL_ID}...")
    
    # Simple handshake test
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents="Are you online?"
    )
    print(f"✅ SUCCESS! Response: {response.text}")

except Exception as e:
    print(f"\n❌ Error connecting to Gemini: {e}")
    exit()

# --- CONTEXT MEMORY ---
class SessionContext:
    def __init__(self):
        self.last_code = None
        self.last_log = None

context = SessionContext()

# --- VAD LOADING ---
try:
    print("[System] Loading VAD model...")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    (get_speech_ts, _, model_class, _, _) = utils
except Exception as e:
    print(f"❌ Error loading VAD: {e}")
    exit()

# --- STATE ---
audio_buffer = bytearray()
lock = threading.Lock()
client_cmd_socket = None

def int16_to_float32(audio_int16):
    audio_np = np.frombuffer(audio_int16, dtype=np.int16)
    return torch.from_numpy(audio_np.astype(np.float32) / 32768.0)

def generate_and_send_code(audio_data):
    """Sends audio + Context (logs/prev code) to Gemini."""
    global client_cmd_socket, context, client
    
    print("[Gemini] 🧠 Processing audio command...")

    # 1. Prepare Audio (WAV container is safer for models)
    f = io.BytesIO()
    with wave.open(f, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)
    f.seek(0)
    audio_bytes = f.read()

    # 2. Construct Contextual Prompt
    context_str = ""
    if context.last_code:
        context_str += f"\n--- PREVIOUS CODE ---\n{context.last_code}\n"
    if context.last_log:
        context_str += f"\n--- PREVIOUS EXECUTION LOGS (User Feedback) ---\n{context.last_log}\n"

    system_instruction = (
        "You are a Nao robot controller (Python 2.7, naoqi). "
        "The user will give you a voice command. "
        "If they are referring to the previous code or errors, MODIFY the previous code to fix or improve it. "
        "If they give a completely new command, ignore the context and write new code. "
        "OUTPUT: ONLY raw Python code. No markdown blocks."
    )

    try:
        # 3. Create Content Parts (New SDK Style)
        prompt_parts = [
            types.Part(text=system_instruction),
            types.Part(text=context_str),
            types.Part(text="User Audio Command:"),
            types.Part(inline_data={
                "mime_type": "audio/wav",
                "data": audio_bytes
            })
        ]

        # 4. Generate Content (The Fix: Use 'client.models', not 'model')
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[types.Content(parts=prompt_parts)]
        )
        
        code = response.text.strip()
        
        # Cleanup formatting (Markdown stripping)
        if code.startswith("```python"): code = code[9:]
        if code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
        
        # Update Context
        context.last_code = code
        context.last_log = None # Clear logs until we get new ones

        print(f"[Gemini] 📜 Generated Code:\n{code[:100]}... (truncated)")

        if client_cmd_socket:
            try:
                payload = code.encode('utf-8')
                header = struct.pack(">L", len(payload))
                client_cmd_socket.sendall(header + payload)
                print("[Server] 🚀 Code sent to robot.")
            except BrokenPipeError:
                print("[Server] ❌ Connection lost while sending code.")
                client_cmd_socket = None
        else:
            print("[Server] ❌ No robot connected to receive code.")

    except Exception as e:
        print(f"[Gemini] ❌ Generation Error: {e}")

def handle_feedback_rx():
    """Receives execution logs/errors from Robot."""
    global context
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((HOST, PORT_FEEDBACK_RX))
        s.listen(1)
        print(f"[Feedback RX] Listening on {PORT_FEEDBACK_RX}")
        
        while True:
            conn, addr = s.accept()
            # print(f"[Feedback RX] Connected by {addr}")
            try:
                while True:
                    raw_len = conn.recv(4)
                    if not raw_len: break
                    msg_len = struct.unpack(">L", raw_len)[0]
                    
                    log_data = b''
                    while len(log_data) < msg_len:
                        packet = conn.recv(msg_len - len(log_data))
                        if not packet: break
                        log_data += packet
                    
                    log_str = log_data.decode('utf-8')
                    print(f"\n[Robot Feedback] ⚠️:\n{log_str}\n")
                    
                    context.last_log = log_str
            except Exception as e:
                print(f"[Feedback RX] Connection Error: {e}")
            finally:
                conn.close()
    except Exception as e:
        print(f"[Feedback RX] Bind Error: {e}")
    finally:
        s.close()

def handle_audio_rx():
    """Receives audio, runs VAD, triggers Gemini."""
    global lock, audio_buffer 

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        s.bind((HOST, PORT_AUDIO_RX))
        s.listen(1)
        print(f"[Audio RX] Listening on {PORT_AUDIO_RX}")

        while True:
            conn, addr = s.accept()
            # print(f"[Audio RX] Connected by {addr}")
            
            # VAD State per connection
            vad_iterator_buffer = bytearray()
            is_speaking = False
            silence_start_time = None
            CHUNKS_PER_VAD = 512 * 2 
            
            try:
                while True:
                    data = conn.recv(4096)
                    if not data: break
                    
                    with lock:
                        audio_buffer.extend(data)
                    
                    vad_iterator_buffer.extend(data)
                    
                    while len(vad_iterator_buffer) >= CHUNKS_PER_VAD:
                        chunk = vad_iterator_buffer[:CHUNKS_PER_VAD]
                        vad_iterator_buffer = vad_iterator_buffer[CHUNKS_PER_VAD:]
                        
                        tensor = int16_to_float32(chunk)
                        
                        if vad_model(tensor, 16000).item() > VAD_THRESHOLD:
                            if not is_speaking:
                                print("[VAD] 🗣️ User started speaking...")
                                is_speaking = True
                            silence_start_time = None
                        else:
                            if is_speaking:
                                if silence_start_time is None:
                                    silence_start_time = time.time()
                                elif (time.time() - silence_start_time) > SILENCE_DURATION_LIMIT:
                                    print("[VAD] 🤫 Silence detected. Processing...")
                                    is_speaking = False
                                    
                                    with lock:
                                        full_audio = bytes(audio_buffer)
                                        audio_buffer = bytearray() 
                                    
                                    threading.Thread(target=generate_and_send_code, args=(full_audio,)).start()
            except Exception as e:
                print(f"[Audio RX] Connection Error: {e}")
            finally:
                conn.close()
    except Exception as e:
        print(f"[Audio RX] Server Error: {e}")
    finally:
        s.close()

def handle_command_tx():
    """Maintains connection to robot for sending code."""
    global client_cmd_socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((HOST, PORT_CMD_TX))
        s.listen(1)
        print(f"[Command TX] Listening on {PORT_CMD_TX}")
        
        while True:
            conn, addr = s.accept()
            print(f"[Command TX] Robot connected from {addr}")
            client_cmd_socket = conn
            
            # Keep-alive loop
            try:
                while True: 
                    time.sleep(1)
                    conn.send(b'') # Heartbeat
            except BrokenPipeError:
                print("[Command TX] Robot disconnected.")
            finally:
                client_cmd_socket = None
                conn.close()
    except Exception as e:
        print(f"[Command TX] Error: {e}")

if __name__ == "__main__":
    # Start all threads
    threading.Thread(target=handle_audio_rx, daemon=True).start()
    threading.Thread(target=handle_command_tx, daemon=True).start()
    threading.Thread(target=handle_feedback_rx, daemon=True).start()
    
    print("✅ All servers running. Press Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")