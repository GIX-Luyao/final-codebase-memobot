#!/usr/bin/env python3
import sys
import time
import queue
import cv2
import struct
import socket
import threading
import asyncio
import numpy as np
import wave
import subprocess
from pathlib import Path
from datetime import datetime

# --- IMPORT ORIGINAL SERVER LOGIC ---
# We import everything from your original file
import mac_master_v3 as original_server

# --- CONFIGURATION ---
ROBOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROBOT_DIR.parent
DATA_DIR = PROJECT_ROOT / "ingest_pipeline" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RECOGNIZED_USER_FRAMES_DIR = ROBOT_DIR / "recognized_user_frames"
RECOGNIZED_USER_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
# Cosine distance threshold: above this = low confidence (reject).
MAX_RECOGNITION_DISTANCE = 0.5
# Wait this long after video connect before taking a frame for recognition (camera exposure stabilizes).
RECOGNITION_FRAME_DELAY = 0.5

# Ensure project root and query_pipeline are on path for recognize_user
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "query_pipeline") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "query_pipeline"))

# --- RECORDER ENGINE ---

class ClipRecorder:
    """Records video + stereo audio (left=user mic, right=TTS) only after recording_started is True."""
    def __init__(self, fps=20, sample_rate=16000, clip_duration=30):
        self.fps = fps
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.video_buffer = []
        self.user_audio_chunks = []   # list of (timestamp, bytes)
        self.tts_audio_chunks = []    # list of (timestamp, bytes)
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.recording_started = False  # Set True only after websocket session is connected

    def start_recording(self):
        with self.lock:
            self.recording_started = True
            self.start_time = time.time()
        print("[Recorder] Recording started (websocket connected).")

    def add_video(self, frame):
        if not self.recording_started:
            return
        with self.lock:
            self.video_buffer.append(frame)

    def add_audio(self, audio_bytes):
        """User mic audio (from robot)."""
        if not self.recording_started:
            return
        with self.lock:
            self.user_audio_chunks.append((time.time(), audio_bytes))

    def add_tts_audio(self, audio_bytes):
        """TTS / response audio (sent to robot)."""
        if not self.recording_started:
            return
        with self.lock:
            self.tts_audio_chunks.append((time.time(), audio_bytes))

    def check_and_save(self):
        if not self.recording_started:
            return
        if time.time() - self.start_time >= self.clip_duration:
            with self.lock:
                v_frames = list(self.video_buffer)
                user_chunks = list(self.user_audio_chunks)
                tts_chunks = list(self.tts_audio_chunks)
                clip_start = self.start_time
                self.video_buffer = []
                self.user_audio_chunks = []
                self.tts_audio_chunks = []
                self.start_time = time.time()
            if v_frames:
                threading.Thread(
                    target=self._finalize_clip,
                    args=(v_frames, user_chunks, tts_chunks, clip_start),
                    daemon=True,
                ).start()

    def _finalize_clip(self, frames, user_chunks, tts_chunks, clip_start):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        v_tmp = str(DATA_DIR / f"v_tmp_{ts}.mp4")
        a_tmp = str(DATA_DIR / f"a_tmp_{ts}.wav")
        final_filename = f"clip_{ts}.mp4"
        final_path = DATA_DIR / final_filename

        total_samples = int(self.clip_duration * self.sample_rate)
        # Build stereo: left = user mic, right = TTS
        left_ch = np.zeros(total_samples, dtype=np.int16)
        right_ch = np.zeros(total_samples, dtype=np.int16)
        bytes_per_sample = 2

        for t, data in user_chunks:
            offset = int((t - clip_start) * self.sample_rate)
            n = min(len(data) // bytes_per_sample, max(0, total_samples - offset))
            if offset < 0:
                data = data[-offset * bytes_per_sample:]
                offset = 0
                n = min(len(data) // bytes_per_sample, total_samples)
            if n > 0:
                left_ch[offset : offset + n] = np.frombuffer(data, dtype=np.int16, count=n)

        for t, data in tts_chunks:
            offset = int((t - clip_start) * self.sample_rate)
            n = min(len(data) // bytes_per_sample, max(0, total_samples - offset))
            if offset < 0:
                data = data[-offset * bytes_per_sample:]
                offset = 0
                n = min(len(data) // bytes_per_sample, total_samples)
            if n > 0:
                right_ch[offset : offset + n] = np.frombuffer(data, dtype=np.int16, count=n)

        stereo_interleaved = np.column_stack((left_ch, right_ch)).astype(np.int16).flatten(order="C")
        with wave.open(a_tmp, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(stereo_interleaved.tobytes())

        # Save Raw Video
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(v_tmp, fourcc, self.fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()

        # Mux with FFmpeg
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", v_tmp, "-i", a_tmp,
                "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
                str(final_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        Path(v_tmp).unlink(missing_ok=True)
        Path(a_tmp).unlink(missing_ok=True)
        print(f"✅ Clip Saved: {final_filename} (stereo: L=user, R=TTS)")

recorder = ClipRecorder()

# --- USER RECOGNITION FROM FIRST FRAME ---

def recognize_user_from_frame(frame):
    """Save frame to a temp image, call recognize_user, return person dict or None.
    Saves the frame to recognized_user_frames/ only when confidence is high enough."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_frame_path = DATA_DIR / f"first_frame_{ts}.png"
    cv2.imwrite(str(first_frame_path), frame)
    try:
        from recognize_user import recognize_user
        person = recognize_user(image_path=str(first_frame_path))
        first_frame_path.unlink(missing_ok=True)
        if person is None:
            return None
        name = person.get("name", "Unknown").replace(" ", "_")
        distance = person.get("distance")
        low_confidence = distance is not None and distance > MAX_RECOGNITION_DISTANCE
        if low_confidence:
            print(f"[Recognition] ERROR: Confidence too low (distance={distance:.4f} > {MAX_RECOGNITION_DISTANCE})")
            saved_path = RECOGNIZED_USER_FRAMES_DIR / f"{ts}_{name}_low_confidence.png"
        else:
            saved_path = RECOGNIZED_USER_FRAMES_DIR / f"{ts}_{name}.png"
        cv2.imwrite(str(saved_path), frame)
        print(f"[Recognition] Saved frame to {saved_path}")
        if low_confidence:
            return None
        return person
    except SystemExit:
        first_frame_path.unlink(missing_ok=True)
        return None
    except Exception as e:
        print(f"[Recognition] Error: {e}")
        first_frame_path.unlink(missing_ok=True)
        return None

# --- OVERRIDDEN THREAD FUNCTIONS ---

def patched_receive_audio_from_robot():
    """Modified version of original audio RX that feeds the recorder (user mic)."""
    print(f"[Patched Audio RX] Listening on {original_server.HOST}:{original_server.PORT_AUDIO_RX}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((original_server.HOST, original_server.PORT_AUDIO_RX))
    server_socket.listen(1)

    conn, addr = server_socket.accept()
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            recorder.add_audio(data)
            if original_server.use_realtime:
                if original_server.agent_instance and not original_server.agent_instance.is_robot_speaking():
                    original_server.agent_instance.ingest_robot_audio(data)
    finally:
        conn.close()
        server_socket.close()


def patched_send_audio_to_robot():
    """Sends audio TO robot (from queue) and feeds TTS into recorder for stereo clip."""
    print(f"[Patched Audio TX] Listening on {original_server.HOST}:{original_server.PORT_AUDIO_TX}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((original_server.HOST, original_server.PORT_AUDIO_TX))
    server_socket.listen(1)

    conn, addr = server_socket.accept()
    try:
        while True:
            data_to_send = None
            if original_server.use_realtime:
                try:
                    data_to_send = original_server.audio_tx_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
            if data_to_send:
                recorder.add_tts_audio(data_to_send)
                conn.sendall(data_to_send)
    except Exception as e:
        print(f"[Patched Audio TX] Error: {e}")
    finally:
        conn.close()
        server_socket.close()

# --- MAIN OVERRIDE ---

async def main():
    # Use original parser logic
    original_server.loop_instance = asyncio.get_running_loop()

    # When --realtime: we defer creating the agent until after we recognize the user from the first frame
    import os
    if "--realtime" in sys.argv:
        original_server.use_realtime = True
        original_server.agent_instance = None  # Created after first frame + recognition
    else:
        original_server.use_realtime = False
        original_server.agent_instance = None

    # Start Audio threads: patched RX (user mic) + patched TX (TTS to robot + recorder)
    threading.Thread(target=patched_receive_audio_from_robot, daemon=True).start()
    threading.Thread(target=patched_send_audio_to_robot, daemon=True).start()

    # Video Setup
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    video_sock.bind((original_server.HOST, original_server.PORT_VIDEO_RX))
    video_sock.listen(1)

    print(f"[Video RX] Recording & Listening on {original_server.PORT_VIDEO_RX}...")
    conn, addr = video_sock.accept()
    payload_size = struct.calcsize(">L")

    # --- Wait for camera to stabilize (~0.5s), then use next frame for recognition ---
    delay_start = time.time()
    first_frame = None
    while True:
        packed_size = original_server.recv_exact(conn, payload_size)
        if not packed_size:
            video_sock.close()
            cv2.destroyAllWindows()
            return
        msg_size = struct.unpack(">L", packed_size)[0]
        frame_data = original_server.recv_exact(conn, msg_size)
        if not frame_data:
            video_sock.close()
            cv2.destroyAllWindows()
            return
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None and (time.time() - delay_start) >= RECOGNITION_FRAME_DELAY:
            first_frame = frame
            break

    if first_frame is None:
        print("[Video RX] Failed to get a frame after delay.")
    else:
        print(f"[Video RX] Frame received after {RECOGNITION_FRAME_DELAY}s. Recognizing user...")
        person = recognize_user_from_frame(first_frame)
        if person:
            name = person.get("name", "Unknown")
            print(f"[Recognized] User: {name}")
        else:
            name = None
            print("[Recognized] No user matched (will run with no user name).")

        if original_server.use_realtime:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                print(f"[System] Starting REALTIME API mode (User: {name or 'Unknown'})")
                original_server.agent_instance = original_server.ServerRealtimeAgent(api_key, user_name=name)
                asyncio.create_task(original_server.agent_instance.run())
            else:
                print("ERROR: OPENAI_API_KEY not found. Realtime mode skipped.")

    # Only start clipping after websocket session is connected (realtime mode)
    recording_started_once = False

    try:
        while True:
            await asyncio.sleep(0.001)

            # Start recorder only after websocket session.created (realtime) or keep off in non-realtime
            if not recording_started_once and original_server.use_realtime and original_server.agent_instance:
                if getattr(original_server.agent_instance, "session_connected", False):
                    recorder.start_recording()
                    recording_started_once = True

            recorder.check_and_save()

            packed_size = original_server.recv_exact(conn, payload_size)
            if not packed_size:
                break
            msg_size = struct.unpack(">L", packed_size)[0]
            frame_data = original_server.recv_exact(conn, msg_size)
            if not frame_data:
                break

            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                recorder.add_video(frame)
                cv2.imshow("NAO Stream (Recording Active)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                if original_server.agent_instance:
                    original_server.agent_instance.is_recording = False
                break
    finally:
        video_sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())