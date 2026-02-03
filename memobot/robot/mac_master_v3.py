#!/usr/bin/env python3
"""
mac_maste_v3.py WITH DENOISING AND RESAMPLING

Server running on Mac.
Handles:
1. Video stream from Robot (Display on Mac)
2. Audio stream from Robot (Play on Mac OR Send to OpenAI Realtime API)
3. Audio stream to Robot (From Mac Mic OR From OpenAI Realtime API)

UPDATES:
- Fixed Noise: Added Realtime Bandpass Filter (300Hz - 4000Hz) to remove fan rumble and hiss.
- Fixed Underrun: Jitter Buffer enabled.
- Fixed Volume: Input Gain enabled.
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

# Try importing Scipy for Filtering
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    print("[Warning] 'scipy' not found. Noise filtering disabled.")
    print("Run: pip install scipy")
    SCIPY_AVAILABLE = False

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT_AUDIO_RX = 50005
PORT_VIDEO_RX = 50006
PORT_AUDIO_TX = 50007

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
ROBOT_RATE = 16000  # Nao Robot native rate
AI_RATE = 24000     # OpenAI Realtime API rate
BYTES_PER_SAMPLE = 2

# DSP CONFIG
INPUT_GAIN = 3.0           # Boost volume (3x)
FILTER_LOW_CUT = 300       # Remove rumble below 300Hz
FILTER_HIGH_CUT = 4000     # Remove hiss above 4000Hz
MIN_SEND_BUFFER_SIZE = 4096 
PLAYBACK_PADDING = 0.5 

# --- GLOBAL STATE ---
audio_tx_queue = queue.Queue()
agent_instance = None
loop_instance = None
use_realtime = False


# --- DSP UTILITIES ---

class RealtimeFilter:
    """
    Maintains filter state (zi) between audio chunks to prevent clicking artifacts.
    """
    def __init__(self, low_cut, high_cut, fs, order=5):
        self.enabled = SCIPY_AVAILABLE
        if not self.enabled:
            return
            
        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        
        # Design Butterworth Bandpass Filter
        self.b, self.a = signal.butter(order, [low, high], btype='band')
        
        # Initialize filter state (zi)
        self.zi = signal.lfilter_zi(self.b, self.a)

    def process(self, audio_bytes):
        if not self.enabled:
            return audio_bytes
        
        # Convert to Float32 for processing
        data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        # Apply Filter with State preservation
        filtered_data, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi)
        
        # Convert back to Int16
        return filtered_data.astype(np.int16).tobytes()


def resample_audio(audio_bytes, src_rate, dst_rate):
    """Resample audio bytes from src_rate to dst_rate using numpy."""
    if src_rate == dst_rate:
        return audio_bytes
    
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    src_len = len(audio_data)
    dst_len = int(src_len * dst_rate / src_rate)
    
    src_x = np.linspace(0, src_len, src_len)
    dst_x = np.linspace(0, src_len, dst_len)
    
    resampled_data = np.interp(dst_x, src_x, audio_data).astype(np.int16)
    return resampled_data.tobytes()

def apply_gain(audio_bytes, gain):
    """Multiply audio signal by gain factor to boost volume."""
    if gain == 1.0:
        return audio_bytes
    
    data = np.frombuffer(audio_bytes, dtype=np.int16)
    boosted = data * gain
    boosted = np.clip(boosted, -32768, 32767)
    return boosted.astype(np.int16).tobytes()


class ServerRealtimeAgent(RealtimeAgent):
    """
    Subclass of RealtimeAgent that overrides hardware I/O.
    """
    def __init__(self, api_key, user_name=None, person_id=None):
        super().__init__(api_key, user_name=user_name, person_id=person_id)
        self.output_stream = None
        self.input_stream = None
        self.playback_end_time = 0.0
        self.state_lock = threading.Lock()
        self.out_buffer = bytearray()
        self.session_connected = False  # Set True when websocket session.created
        # Initialize Noise Filter
        self.noise_filter = RealtimeFilter(FILTER_LOW_CUT, FILTER_HIGH_CUT, ROBOT_RATE)

    def register_audio_payload(self, audio_bytes_at_robot_rate):
        bytes_per_sec = ROBOT_RATE * BYTES_PER_SAMPLE * CHANNELS
        duration = len(audio_bytes_at_robot_rate) / float(bytes_per_sec)
        current_time = time.time()
        with self.state_lock:
            if self.playback_end_time > current_time:
                self.playback_end_time += duration
            else:
                self.playback_end_time = current_time + duration

    def is_robot_speaking(self):
        with self.state_lock:
            return time.time() < (self.playback_end_time + PLAYBACK_PADDING)

    async def capture_audio(self):
        print("[Agent] Ready to receive audio from Robot socket...")
        while self.is_recording:
            await asyncio.sleep(0.1)

    async def receive_messages(self):
        print("[Agent] Listening for API responses...")
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type", "")

                if event_type == "session.created":
                    self.session_connected = True
                    print("🎙️  Session started. Robot is listening...")
                
                elif event_type == "response.audio.delta":
                    audio_b64 = event.get("delta", "")
                    if audio_b64:
                        audio_24k = base64.b64decode(audio_b64)
                        audio_16k = resample_audio(audio_24k, src_rate=AI_RATE, dst_rate=ROBOT_RATE)
                        self.out_buffer.extend(audio_16k)

                        if len(self.out_buffer) >= MIN_SEND_BUFFER_SIZE:
                            chunk = bytes(self.out_buffer)
                            self.out_buffer.clear()
                            self.register_audio_payload(chunk)
                            audio_tx_queue.put(chunk)

                elif event_type == "response.audio.done":
                    if len(self.out_buffer) > 0:
                        chunk = bytes(self.out_buffer)
                        self.out_buffer.clear()
                        self.register_audio_payload(chunk)
                        audio_tx_queue.put(chunk)

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
        Pipeline: 
        1. Denoise (Bandpass Filter)
        2. Amplify (Gain)
        3. Resample (16k -> 24k)
        4. Send to AI
        """
        if self.ws:
            # 1. Bandpass Filter (Removes Fan Rumble + Hiss)
            clean_audio = self.noise_filter.process(audio_bytes_16k)

            # 2. Apply Gain (Boost Volume)
            boosted_audio = apply_gain(clean_audio, INPUT_GAIN)

            # 3. Resample 16kHz -> 24kHz for OpenAI
            audio_24k = resample_audio(boosted_audio, src_rate=ROBOT_RATE, dst_rate=AI_RATE)
            
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
                # Echo Gate
                if agent_instance and agent_instance.is_robot_speaking():
                    continue 
                
                # Ingest (Filtering -> Gain -> Resample)
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
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=ROBOT_RATE, input=True, frames_per_buffer=1024)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT_AUDIO_TX))
    server_socket.listen(1)

    conn = None
    try:
        conn, addr = server_socket.accept()
        print(f"[Audio TX] Robot Connected: {addr}")

        while True:
            data_to_send = None

            if use_realtime:
                try:
                    data_to_send = audio_tx_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
            elif stream:
                try:
                    data_to_send = stream.read(1024, exception_on_overflow=False)
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
        print(f"[System] Bandpass Filter Enabled (300Hz-4000Hz)")
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



