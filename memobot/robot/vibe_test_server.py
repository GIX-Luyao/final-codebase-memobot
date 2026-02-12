#!/usr/bin/env python3
import socket
import struct
import threading
import os
import io
import time
import wave
import numpy as np
import torch
import google.generativeai as genai
from dotenv import load_dotenv
import os




# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT_AUDIO_RX    = 50005  # Robot Mic -> Server
PORT_CMD_TX      = 50007  # Server Code -> Robot
PORT_FEEDBACK_RX = 50008  # Robot Logs -> Server (NEW)

# VAD Settings
VAD_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
SILENCE_DURATION_LIMIT = 1.0 

# Gemini Setup
load_dotenv(override=True)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash') 

# --- CONTEXT MEMORY ---
class SessionContext:
    def __init__(self):
        self.last_code = None
        self.last_log = None

context = SessionContext()

# --- VAD LOADING (Same as before) ---
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    (get_speech_ts, _, model_class, _, _) = utils
except:
    pass # Assume loaded or handled

# --- STATE ---
audio_buffer = bytearray()
lock = threading.Lock()
client_cmd_socket = None

def int16_to_float32(audio_int16):
    audio_np = np.frombuffer(audio_int16, dtype=np.int16)
    return torch.from_numpy(audio_np.astype(np.float32) / 32768.0)

def generate_and_send_code(audio_data):
    """Sends audio + Context (logs/prev code) to Gemini."""
    global client_cmd_socket, context
    
    print("[Gemini] 🧠 Processing audio command...")

    # 1. Prepare Audio
    f = io.BytesIO()
    with wave.open(f, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)
    f.seek(0)
    audio_blob = {"mime_type": "audio/wav", "data": f.read()}

    # 2. Construct Contextual Prompt
    # This is the "Smooth Interaction" magic. We give the AI the full picture.
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
        "OUTPUT: ONLY raw Python code. No markdown."
    )

    prompt_parts = [system_instruction, context_str, "User Audio:", audio_blob]

    try:
        response = model.generate_content(prompt_parts)
        code = response.text.strip()
        
        # Cleanup formatting
        if code.startswith("```python"): code = code[9:]
        if code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
        
        # Update Context
        context.last_code = code
        context.last_log = None # Clear logs until we get new ones

        print(f"[Gemini] 📜 Generated Code:\n{code[:100]}... (truncated)")

        if client_cmd_socket:
            payload = code.encode('utf-8')
            header = struct.pack(">L", len(payload))
            client_cmd_socket.sendall(header + payload)
            print("[Server] 🚀 Code sent to robot.")
        else:
            print("[Server] ❌ No robot connected.")

    except Exception as e:
        print(f"[Gemini] Error: {e}")

def handle_feedback_rx():
    """Receives execution logs/errors from Robot."""
    global context
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT_FEEDBACK_RX))
    s.listen(1)
    print(f"[Feedback RX] Listening on {PORT_FEEDBACK_RX}")
    
    conn, _ = s.accept()
    try:
        while True:
            # Protocol: [Length (4 bytes)] [Log String]
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
            
            # Save this log to context so Gemini sees it next turn
            context.last_log = log_str
            
    finally:
        conn.close()
        s.close()

# --- OTHER THREADS (Audio RX, CMD TX) ---
# (Keep handle_audio_rx and handle_command_tx from previous code exactly the same)
# ... [Paste handle_audio_rx and handle_command_tx here] ...

def handle_audio_rx():
    # ... (Same as previous provided code) ...
    # BUT: Use the updated generate_and_send_code function above
    # Copy/Paste the VAD logic from previous response
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT_AUDIO_RX))
    s.listen(1)
    conn, _ = s.accept()
    vad_iterator_buffer = bytearray()
    is_speaking = False
    silence_start_time = 0
    CHUNKS_PER_VAD = 512 * 2
    try:
        while True:
            data = conn.recv(4096)
            if not data: break
            with lock: audio_buffer.extend(data)
            vad_iterator_buffer.extend(data)
            while len(vad_iterator_buffer) >= CHUNKS_PER_VAD:
                chunk = vad_iterator_buffer[:CHUNKS_PER_VAD]
                vad_iterator_buffer = vad_iterator_buffer[CHUNKS_PER_VAD:]
                tensor = int16_to_float32(chunk)
                if vad_model(tensor, 16000).item() > VAD_THRESHOLD:
                    if not is_speaking:
                        print("[VAD] 🗣️ Speaking...")
                        is_speaking = True
                        with lock: pass 
                    silence_start_time = None
                else:
                    if is_speaking:
                        if silence_start_time is None: silence_start_time = time.time()
                        elif (time.time() - silence_start_time) > SILENCE_DURATION_LIMIT:
                            print("[VAD] 🤫 Silence. Processing...")
                            is_speaking = False
                            with lock:
                                full_audio = bytes(audio_buffer)
                                audio_buffer = bytearray()
                            threading.Thread(target=generate_and_send_code, args=(full_audio,)).start()
    finally: conn.close()

def handle_command_tx():
    # ... (Same as previous provided code) ...
    global client_cmd_socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT_CMD_TX))
    s.listen(1)
    while True:
        try:
            conn, _ = s.accept()
            client_cmd_socket = conn
            while True: 
                time.sleep(1); conn.send(b'')
        except: client_cmd_socket = None

if __name__ == "__main__":
    threading.Thread(target=handle_audio_rx, daemon=True).start()
    threading.Thread(target=handle_command_tx, daemon=True).start()
    threading.Thread(target=handle_feedback_rx, daemon=True).start() # NEW THREAD
    while True: time.sleep(1)