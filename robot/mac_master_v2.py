#!/usr/bin/env python3
"""
mac_master.py

Server running on Mac.
Handles:
1. Video stream from Robot (Display on Mac)
2. Audio stream from Robot (Play on Mac OR Send to OpenAI Realtime API)
3. Audio stream to Robot (From Mac Mic OR From OpenAI Realtime API)

UPDATES:
- Fixed "Underrun": Sends continuous SILENCE frames to robot when AI is not speaking.
- Fixed "Low Volume": Applies DIGITAL GAIN (4x) to user audio before sending to AI.
"""

import socket
import threading
import pyaudio
import cv2
import numpy as np
import struct
import sys
import time
import asyncio
import queue
import argparse
import base64
import json
from pathlib import Path

# --- IMPORTS FROM PROJECT ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from query_pipeline.robo_client import RealtimeAgent
except ImportError:
    print("[Error] Could not import RealtimeAgent from robo_client.py")
    sys.exit(1)

try:
    from Memobot import MemobotService
except ImportError:
    print("[Warning] Memobot package not found.")
    MemobotService = None

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT_AUDIO_RX = 50005
PORT_VIDEO_RX = 50006
PORT_AUDIO_TX = 50007

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1

# RATES
ROBOT_RATE = 16000  # Nao Robot native rate
AI_RATE = 24000     # OpenAI Realtime API rate
CHUNK = 1024        # Frames per buffer
BYTES_PER_SAMPLE = 2

# VOLUME & TIMING CONFIG
MIC_GAIN = 4.0          # Multiplier: Amplify User Voice 4x (Adjust if too loud/distorted)
PLAYBACK_PADDING = 0.5  # Block mic for 0.5s after AI finishes speaking

# --- GLOBAL STATE ---
audio_tx_queue = queue.Queue()
agent_instance = None
loop_instance = None
use_realtime = False

def resample_audio(audio_bytes, src_rate, dst_rate):
    """Resample audio bytes using numpy linear interpolation."""
    if src_rate == dst_rate:
        return audio_bytes
    
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    src_len = len(audio_data)
    dst_len = int(src_len * dst_rate / src_rate)
    
    src_x = np.linspace(0, src_len, src_len)
    dst_x = np.linspace(0, src_len, dst_len)
    
    resampled_data = np.interp(dst_x, src_x, audio_data).astype(np.int16)
    return resampled_data.tobytes()

def amplify_audio(audio_bytes, gain):
    """Amplify audio by a multiplier and clip to int16 range."""
    if gain == 1.0:
        return audio_bytes
        
    data = np.frombuffer(audio_bytes, dtype=np.int16)
    # Multiply
    amplified = data * gain
    # Clip to prevent overflow distortion
    amplified = np.clip(amplified, -32768, 32767)
    return amplified.astype(np.int16).tobytes()


class ServerRealtimeAgent(RealtimeAgent):
    """
    Subclass of RealtimeAgent that overrides hardware I/O.
    """
    def __init__(self, api_key, user_name=None):
        super().__init__(api_key, user_name)
        self.output_stream = None
        self.input_stream = None
        
        # Virtual Playback Tracking
        self.playback_end_time = 0.0
        self.state_lock = threading.Lock()

    def register_audio_payload(self, audio_bytes_at_robot_rate):
        """
        Calculate how long this chunk of audio will take to play ON THE ROBOT
        and update the blocked-until timestamp.
        """
        bytes_per_sec = ROBOT_RATE * BYTES_PER_SAMPLE * CHANNELS
        duration = len(audio_bytes_at_robot_rate) / float(bytes_per_sec)
        
        current_time = time.time()
        
        with self.state_lock:
            if self.playback_end_time > current_time:
                self.playback_end_time += duration
            else:
                self.playback_end_time = current_time + duration

    def is_robot_speaking(self):
        """Check if we are currently within the virtual playback window."""
        with self.state_lock:
            return time.time() < (self.playback_end_time + PLAYBACK_PADDING)

    async def capture_audio(self):
        """Override: Keep alive, but do not capture local mic."""
        print("[Agent] Ready to receive audio from Robot socket...")
        while self.is_recording:
            await asyncio.sleep(0.1)

    async def receive_messages(self):
        """Override: Push API audio to queue instead of playing locally."""
        print("[Agent] Listening for API responses...")
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type", "")

                if event_type == "session.created":
                    print("🎙️  Session started. Robot is listening...")
                
                elif event_type == "response.audio.delta":
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        # 1. Decode (24kHz)
                        audio_24k = base64.b64decode(audio_b64)
                        
                        # 2. Resample 24kHz -> 16kHz for Robot
                        audio_16k = resample_audio(audio_24k, src_rate=AI_RATE, dst_rate=ROBOT_RATE)
                        
                        # 3. Update Echo Gate (Using robot rate duration)
                        self.register_audio_payload(audio_16k)
                        
                        # 4. Queue for sending
                        audio_tx_queue.put(audio_16k)

                elif event_type == "response.audio_transcript.delta":
                    print(event.get("delta", ""), end="", flush=True)
                
                elif event_type == "response.audio_transcript.done":
                    print()
                
                elif event_type == "response.function_call_arguments.done":
                    call_id = event.get("call_id", "")
                    name = event.get("name", "")
                    arguments = event.get("arguments", "{}")
                    await self.handle_function_call(call_id, name, arguments)

        except Exception as e:
            print(f"\n[Agent] Connection error: {e}")

    def ingest_robot_audio(self, audio_bytes_16k):
        """
        Takes 16kHz audio from robot, Amplifies it, Resamples to 24kHz, sends to OpenAI.
        """
        if self.ws:
            # 1. Amplify (Gain)
            audio_loud = amplify_audio(audio_bytes_16k, gain=MIC_GAIN)

            # 2. Resample 16kHz -> 24kHz for OpenAI
            audio_24k = resample_audio(audio_loud, src_rate=ROBOT_RATE, dst_rate=AI_RATE)
            
            asyncio.run_coroutine_threadsafe(self.send_audio(audio_24k), loop_instance)


# --- THREADS ---

def thread_receive_audio_from_robot():
    """Listens on 50005. Receives audio FROM Robot."""
    global agent_instance, loop_instance, use_realtime
    print(f"[Audio RX] Listening on {HOST}:{PORT_AUDIO_RX}...")

    stream = None
    p = None
    if not use_realtime:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=ROBOT_RATE, output=True)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT_AUDIO_RX))
    server_socket.listen(1)

    conn = None
    try:
        conn, addr = server_socket.accept()
        print(f"[Audio RX] Robot Connected: {addr}")

        while True:
            data = conn.recv(4096)
            if not data: break

            if use_realtime:
                # --- ECHO GATE ---
                # Completely ignore input while Robot is speaking
                if agent_instance and agent_instance.is_robot_speaking():
                    continue 
                
                # If gate is open, process input
                if agent_instance and loop_instance:
                    agent_instance.ingest_robot_audio(data)

            elif stream:
                stream.write(data)

    except Exception as e:
        print(f"[Audio RX] Error: {e}")
    finally:
        if conn: conn.close()
        server_socket.close()
        if stream:
            stream.stop_stream()
            stream.close()
        if p: p.terminate()

def thread_send_audio_to_robot():
    """Listens on 50007. Sends audio TO Robot."""
    global agent_instance, use_realtime, audio_tx_queue
    print(f"[Audio TX] Listening on {HOST}:{PORT_AUDIO_TX}...")

    stream = None
    p = None
    if not use_realtime:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=ROBOT_RATE, input=True, frames_per_buffer=CHUNK)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT_AUDIO_TX))
    server_socket.listen(1)

    conn = None
    try:
        conn, addr = server_socket.accept()
        print(f"[Audio TX] Robot Connected: {addr}")

        # Pre-generate a chunk of silence (16kHz, mono, 16-bit)
        # Size = CHUNK * 2 bytes
        silence_frame = b'\x00' * (CHUNK * BYTES_PER_SAMPLE)

        while True:
            data_to_send = None

            if use_realtime:
                try:
                    # Non-blocking check or very short timeout
                    data_to_send = audio_tx_queue.get(timeout=0.02) # 20ms timeout
                except queue.Empty:
                    # --- UNDERRUN FIX: SEND SILENCE ---
                    # If the AI has nothing to say, send zeros.
                    # This keeps the robot's buffer full so it doesn't underrun.
                    data_to_send = silence_frame
            elif stream:
                try:
                    data_to_send = stream.read(CHUNK, exception_on_overflow=False)
                except IOError:
                    continue

            if data_to_send:
                conn.sendall(data_to_send)

    except Exception as e:
        print(f"[Audio TX] Error: {e}")
    finally:
        if conn: conn.close()
        server_socket.close()
        if stream:
            stream.stop_stream()
            stream.close()
        if p: p.terminate()

def recv_exact(sock, n):
    """Helper to receive exactly n bytes."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data += packet
    return data

async def main():
    global agent_instance, loop_instance, use_realtime

    parser = argparse.ArgumentParser()
    parser.add_argument("--realtime", action="store_true", help="Enable OpenAI Realtime API")
    parser.add_argument("--user-name", type=str, default=None, help="User Name")
    args = parser.parse_args()

    use_realtime = args.realtime
    loop_instance = asyncio.get_running_loop()

    if use_realtime:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not found.")
            return
        print(f"[System] Starting in REALTIME API mode (User: {args.user_name})")
        print(f"[System] Input Gain: {MIC_GAIN}x | Resampling: {AI_RATE}<->{ROBOT_RATE}")
        agent_instance = ServerRealtimeAgent(api_key, user_name=args.user_name)
    else:
        print("[System] Starting in STANDARD PASS-THROUGH mode")

    if MemobotService:
        try:
            memobot_service = MemobotService.from_env(group_id='tenant_001')
            await memobot_service.initialize()
        except Exception:
            pass

    try:
        t1 = threading.Thread(target=thread_receive_audio_from_robot)
        t2 = threading.Thread(target=thread_send_audio_to_robot)
        t1.daemon = True
        t2.daemon = True
        t1.start()
        t2.start()

        if use_realtime and agent_instance:
            asyncio.create_task(agent_instance.run())

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT_VIDEO_RX))
        server_socket.listen(1)

        print(f"[Video RX] Listening on {HOST}:{PORT_VIDEO_RX}...")
        conn, addr = server_socket.accept()
        payload_size = struct.calcsize(">L")

        while True:
            await asyncio.sleep(0.001)

            packed_msg_size = recv_exact(conn, payload_size)
            if not packed_msg_size: break
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            frame_data = recv_exact(conn, msg_size)
            if not frame_data: break

            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("NAO Robot Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if agent_instance: agent_instance.is_recording = False
                break

    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        if 'server_socket' in locals(): server_socket.close()
        if 'conn' in locals(): conn.close()
        cv2.destroyAllWindows()
        if 'memobot_service' in locals() and memobot_service: await memobot_service.close()

if __name__ == "__main__":
    asyncio.run(main())
