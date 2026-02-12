#!/usr/bin/env python3
import socket
import struct
import threading
import os
import io
import time
import wave
import numpy as np
import cv2
import torch
import google.generativeai as genai

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT_AUDIO_RX = 50005  # Robot Mic -> Server
PORT_VIDEO_RX = 50006  # Robot Cam -> Server
PORT_CMD_TX   = 50007  # Server Code -> Robot (Replaces Audio TX)

# VAD Settings
VAD_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
SILENCE_DURATION_LIMIT = 1.0  # Seconds of silence to trigger generation

# Gemini Setup
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash') 

# --- VAD LOADING ---
try:
    # Load Silero VAD from Torch Hub (standard method)
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False)
    (get_speech_ts, _, model_class, _, _) = utils
    print("[Server] Silero VAD Loaded.")
except Exception as e:
    print(f"[Server] Error loading VAD: {e}")
    sys.exit(1)

# --- STATE ---
audio_buffer = bytearray()
lock = threading.Lock()
client_cmd_socket = None

def int16_to_float32(audio_int16):
    """Convert int16 audio bytes to float32 numpy array for VAD."""
    audio_np = np.frombuffer(audio_int16, dtype=np.int16)
    return torch.from_numpy(audio_np.astype(np.float32) / 32768.0)

def generate_and_send_code(audio_data):
    """Sends audio to Gemini, gets code, sends to robot."""
    global client_cmd_socket
    
    print("[Gemini] 🧠 Processing audio command...")

    # 1. Save Audio to In-Memory WAV
    try:
        f = io.BytesIO()
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
        f.seek(0)
        audio_blob = {"mime_type": "audio/wav", "data": f.read()}
    except Exception as e:
        print(f"[Error] Audio processing failed: {e}")
        return

    # 2. Prompt Gemini
    # We explicitly ask for raw Python 2.7 code compatible with Naoqi.
    prompt = (
        "You are a robot controller. Listen to the user's voice command and write "
        "a valid Python 2.7 script using the 'naoqi' library to execute it. "
        "Assume 'ALProxy' is imported. "
        "Do not use markdown formatting (```). Return ONLY the raw code string. "
        "Example output: tts = ALProxy('ALTextToSpeech', '127.0.0.1', 9559); tts.say('Hello')"
    )

    try:
        response = model.generate_content([prompt, audio_blob])
        code = response.text.strip()
        
        # Strip markdown if Gemini ignores instruction
        if code.startswith("```python"): code = code[9:]
        if code.startswith("```"): code = code[3:]
        if code.endswith("```"): code = code[:-3]
        
        print(f"[Gemini] 📜 Generated Code:\n{code}")

        # 3. Send to Robot
        if client_cmd_socket:
            payload = code.encode('utf-8')
            # Packet: [Length (4 bytes)] [Payload]
            header = struct.pack(">L", len(payload))
            client_cmd_socket.sendall(header + payload)
            print("[Server] 🚀 Code sent to robot.")
        else:
            print("[Server] ❌ No robot connected to receive code.")

    except Exception as e:
        print(f"[Gemini] Error: {e}")


def handle_audio_rx():
    """Receives audio, runs VAD, triggers Gemini."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT_AUDIO_RX))
    s.listen(1)
    print(f"[Audio RX] Listening on {PORT_AUDIO_RX}")

    conn, _ = s.accept()
    
    global audio_buffer
    vad_iterator_buffer = bytearray()
    
    # VAD State
    is_speaking = False
    silence_start_time = 0
    CHUNKS_PER_VAD = 512 * 2 # 512 samples * 2 bytes
    
    try:
        while True:
            data = conn.recv(4096)
            if not data: break
            
            # 1. Buffer for the full command
            with lock:
                audio_buffer.extend(data)
            
            # 2. Run VAD on small chunks
            vad_iterator_buffer.extend(data)
            
            while len(vad_iterator_buffer) >= CHUNKS_PER_VAD:
                chunk = vad_iterator_buffer[:CHUNKS_PER_VAD]
                vad_iterator_buffer = vad_iterator_buffer[CHUNKS_PER_VAD:]
                
                tensor = int16_to_float32(chunk)
                confidence = vad_model(tensor, 16000).item()
                
                if confidence > VAD_THRESHOLD:
                    if not is_speaking:
                        print("[VAD] 🗣️ User started speaking...")
                        is_speaking = True
                        # Clear buffer to start fresh recording
                        with lock:
                            # Keep a tiny bit of context or clear entirely
                            pass 
                    silence_start_time = None
                else:
                    if is_speaking:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif (time.time() - silence_start_time) > SILENCE_DURATION_LIMIT:
                            print("[VAD] 🤫 Silence detected. Processing command...")
                            is_speaking = False
                            
                            # EXTRACT AND PROCESS
                            with lock:
                                full_audio = bytes(audio_buffer)
                                audio_buffer = bytearray() # Reset
                            
                            threading.Thread(target=generate_and_send_code, args=(full_audio,)).start()

    finally:
        conn.close()
        s.close()

def handle_command_tx():
    """Establishes connection to send code back to robot."""
    global client_cmd_socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT_CMD_TX))
    s.listen(1)
    print(f"[Command TX] Waiting for robot on {PORT_CMD_TX}")
    
    while True:
        conn, addr = s.accept()
        print(f"[Command TX] Robot connected from {addr}")
        client_cmd_socket = conn
        # Keep connection alive until it breaks, then wait for reconnect
        try:
            while True:
                time.sleep(1)
                # Simple check if socket is alive
                conn.send(b'') 
        except:
            print("[Command TX] Robot disconnected.")
            client_cmd_socket = None

def handle_video_rx():
    """Optional: View what the robot sees."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT_VIDEO_RX))
    s.listen(1)
    print(f"[Video RX] Listening on {PORT_VIDEO_RX}")
    
    conn, _ = s.accept()
    payload_size = struct.calcsize(">L")
    
    try:
        while True:
            packed_msg_size = b''
            while len(packed_msg_size) < payload_size:
                data = conn.recv(payload_size - len(packed_msg_size))
                if not data: return
                packed_msg_size += data
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            
            frame_data = b''
            while len(frame_data) < msg_size:
                data = conn.recv(msg_size - len(frame_data))
                if not data: return
                frame_data += data
                
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Robot View', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        conn.close()
        s.close()

if __name__ == "__main__":
    t1 = threading.Thread(target=handle_audio_rx, daemon=True)
    t2 = threading.Thread(target=handle_command_tx, daemon=True)
    t3 = threading.Thread(target=handle_video_rx, daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()