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
import mac_master_v3 as original_server

# --- CONFIGURATION ---
ROBOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROBOT_DIR.parent
DATA_DIR = PROJECT_ROOT / "ingest_pipeline" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RECOGNIZED_USER_FRAMES_DIR = ROBOT_DIR / "recognized_user_frames"
RECOGNIZED_USER_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

MAX_RECOGNITION_DISTANCE = 0.5
RECOGNITION_FRAME_DELAY = 0.5
NETWORK_LATENCY_DELAY = 0.25

# Set in main() from --ingest: when True, run ingest pipeline after each 30s clip
INGEST_MEMORIES = False

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "query_pipeline") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "query_pipeline"))

# --- RECORDER ENGINE ---

class ClipRecorder:
    """
    Records video + mixed audio.
    CRITICAL FIXES:
    1. Smart Snapping: Prevents stuttering/overlapping TTS.
    2. SOFTWARE ECHO CANCELLATION (DUCKING): Mutes the microphone track 
       whenever the robot is speaking to prevent 'room echo' in the recording.
    """
    def __init__(self, fps=20, sample_rate=16000, clip_duration=30, network_latency_delay=NETWORK_LATENCY_DELAY):
        self.fps = fps
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.network_latency_delay = network_latency_delay
        
        # Buffers
        self.video_buffer = []
        self.user_audio_bytes = bytearray() # Continuous stream
        self.tts_audio_chunks = []          # (timestamp, bytes)
        
        # State
        self.lock = threading.Lock()
        self.start_time = None
        self.recording_started = False 

    def start_recording(self):
        with self.lock:
            self.recording_started = True
            self.start_time = time.time()
            self.video_buffer = []
            self.user_audio_bytes = bytearray()
            self.tts_audio_chunks = []
        print("[Recorder] 🔴 Recording session started.")

    def add_video(self, frame):
        if not self.recording_started:
            return
        with self.lock:
            self.video_buffer.append(frame)

    def add_audio(self, audio_bytes):
        """User mic audio: Continuous stream."""
        if not self.recording_started:
            return
        with self.lock:
            self.user_audio_bytes.extend(audio_bytes)

    def add_tts_audio(self, audio_bytes):
        """TTS audio: Timestamped events."""
        if not self.recording_started:
            return
        timestamp = time.time()
        with self.lock:
            self.tts_audio_chunks.append((timestamp, audio_bytes))

    def check_and_save(self):
        if not self.recording_started:
            return
        
        if time.time() - self.start_time >= self.clip_duration:
            with self.lock:
                frames_to_save = list(self.video_buffer)
                user_bytes_to_save = bytes(self.user_audio_bytes)
                tts_chunks_to_save = list(self.tts_audio_chunks)
                clip_start_time = self.start_time
                
                self.video_buffer = []
                self.user_audio_bytes = bytearray()
                self.tts_audio_chunks = []
                self.start_time = time.time()

            if frames_to_save:
                threading.Thread(
                    target=self._finalize_clip,
                    args=(frames_to_save, user_bytes_to_save, tts_chunks_to_save, clip_start_time),
                    daemon=True,
                ).start()

    def _finalize_clip(self, frames, user_bytes, tts_chunks, clip_start):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        v_tmp = str(DATA_DIR / f"v_tmp_{ts}.mp4")
        a_tmp = str(DATA_DIR / f"a_tmp_{ts}.wav")
        final_path = str(DATA_DIR / f"clip_{ts}.mp4")

        try:
            total_samples = int(self.clip_duration * self.sample_rate)
            
            # 1. The Mixing Canvas
            mixed_track = np.zeros(total_samples, dtype=np.int32)
            
            # 
            # 2. Create a "Mute Mask" for the User Mic.
            # Starts as all 1.0 (Full Volume). We will set it to 0.0 where Robot speaks.
            mic_ducking_mask = np.ones(total_samples, dtype=np.float32)

            # --- PROCESS TTS (ROBOT) FIRST to build the Mask ---
            tts_chunks.sort(key=lambda x: x[0])
            last_write_end_idx = -1
            SNAP_THRESHOLD_SAMPLES = int(0.2 * self.sample_rate)

            for t, data in tts_chunks:
                audio_arr = np.frombuffer(data, dtype=np.int16)
                chunk_len = len(audio_arr)
                
                # Determine placement (Smart Snapping)
                theoretical_start_sec = (t - clip_start) + self.network_latency_delay
                start_idx = int(theoretical_start_sec * self.sample_rate)
                
                if last_write_end_idx != -1:
                    gap = start_idx - last_write_end_idx
                    if gap < SNAP_THRESHOLD_SAMPLES:
                        start_idx = max(0, last_write_end_idx)
                
                if start_idx >= total_samples:
                    continue

                end_idx = min(total_samples, start_idx + chunk_len)
                write_len = end_idx - start_idx
                
                if write_len > 0:
                    # Write Robot Audio to Mix
                    mixed_track[start_idx:end_idx] += audio_arr[:write_len]
                    
                    # UPDATE MASK: Silence the microphone during this segment
                    # This removes the "echo" of the robot hearing itself.
                    mic_ducking_mask[start_idx:end_idx] = 0.0
                    
                    last_write_end_idx = end_idx

            # --- PROCESS USER (MIC) ---
            user_arr = np.frombuffer(user_bytes, dtype=np.int16)
            samples_to_copy = min(len(user_arr), total_samples)
            
            # Apply the Mask (Echo Cancellation)
            # Where mask is 0.0, user audio becomes silence.
            processed_user_audio = user_arr[:samples_to_copy] * mic_ducking_mask[:samples_to_copy]
            
            # Add to Mix
            mixed_track[:samples_to_copy] += processed_user_audio.astype(np.int32)

            # --- SAVE ---
            final_audio = np.clip(mixed_track, -32768, 32767).astype(np.int16)
            with wave.open(a_tmp, "wb") as wf:
                wf.setnchannels(1) 
                wf.setsampwidth(2) 
                wf.setframerate(self.sample_rate)
                wf.writeframes(final_audio.tobytes())

            if not frames: return
            h, w, _ = frames[0].shape
            out = cv2.VideoWriter(v_tmp, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for f in frames: out.write(f)
            out.release()

            subprocess.run([
                "ffmpeg", "-y", "-i", v_tmp, "-i", a_tmp,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ac", "2",
                "-strict", "experimental", final_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"[Recorder] ✅ Saved Clip: {Path(final_path).name} (Echo Removed)")

            # Run ingest pipeline on the new clip (non-blocking, with status logs) if enabled
            if INGEST_MEMORIES:
                clip_name = Path(final_path).name
                ingest_script = PROJECT_ROOT / "ingest_pipeline" / "main.py"
                repo_root = PROJECT_ROOT.parent
                if ingest_script.exists():
                    def _run_ingest_and_log():
                        print(f"[Ingest] 📤 Starting pipeline for {clip_name}...")
                        proc = subprocess.Popen(
                            [sys.executable, "-u", str(ingest_script), clip_name],
                            cwd=str(repo_root),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        for line in proc.stdout:
                            line = line.rstrip()
                            if line:
                                print(f"[Ingest] {line}")
                        proc.wait()
                        if proc.returncode == 0:
                            print(f"[Ingest] ✅ Done {clip_name} (exit 0)")
                        else:
                            print(f"[Ingest] ❌ Failed {clip_name} (exit {proc.returncode})")
                    threading.Thread(target=_run_ingest_and_log, daemon=True).start()
                else:
                    print(f"[Recorder] ⚠ Ingest script not found: {ingest_script}")

        except Exception as e:
            print(f"[Recorder] ❌ Error: {e}")
        finally:
            Path(v_tmp).unlink(missing_ok=True)
            Path(a_tmp).unlink(missing_ok=True)

recorder = ClipRecorder()

# --- RECOGNITION ---
def recognize_user_from_frame(frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = DATA_DIR / f"first_frame_{ts}.png"
    cv2.imwrite(str(tmp_path), frame)
    try:
        from recognize_user import recognize_user
        person = recognize_user(image_path=str(tmp_path))
        tmp_path.unlink(missing_ok=True)
        if person is None: return None
        
        name = person.get("name", "Unknown").replace(" ", "_")
        dist = person.get("distance", 1.0)
        
        if dist > MAX_RECOGNITION_DISTANCE:
            print(f"[Recognition] Low confidence ({dist:.4f}). Ignoring.")
            cv2.imwrite(str(RECOGNIZED_USER_FRAMES_DIR / f"{ts}_{name}_low_conf.png"), frame)
            return None
            
        cv2.imwrite(str(RECOGNIZED_USER_FRAMES_DIR / f"{ts}_{name}.png"), frame)
        return person
    except Exception as e:
        print(f"[Recognition] Error: {e}")
        tmp_path.unlink(missing_ok=True)
        return None

# --- PATCHED NETWORKING ---

def patched_receive_audio_from_robot():
    """Reads audio from socket in a loop and feeds the continuous buffer."""
    print(f"[Audio RX] Listening on {original_server.PORT_AUDIO_RX}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((original_server.HOST, original_server.PORT_AUDIO_RX))
    sock.listen(1)
    
    conn, _ = sock.accept()
    try:
        while True:
            data = conn.recv(4096)
            if not data: break
            
            recorder.add_audio(data)
            
            if original_server.use_realtime and original_server.agent_instance:
                if not original_server.agent_instance.is_robot_speaking():
                    original_server.agent_instance.ingest_robot_audio(data)
    except Exception as e:
        print(f"[Audio RX] Error: {e}")
    finally:
        conn.close()
        sock.close()

def patched_send_audio_to_robot():
    """Reads from Queue -> Sends to Robot -> Feeds Recorder (Timestamped)."""
    print(f"[Audio TX] Listening on {original_server.PORT_AUDIO_TX}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((original_server.HOST, original_server.PORT_AUDIO_TX))
    sock.listen(1)
    
    conn, _ = sock.accept()
    try:
        while True:
            data = None
            if original_server.use_realtime:
                try:
                    data = original_server.audio_tx_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            
            if data:
                recorder.add_tts_audio(data)
                conn.sendall(data)
    except Exception as e:
        print(f"[Audio TX] Error: {e}")
    finally:
        conn.close()
        sock.close()

# --- MAIN ---

async def main():
    original_server.loop_instance = asyncio.get_running_loop()
    
    # Mode Setup
    import os
    global INGEST_MEMORIES
    INGEST_MEMORIES = "--ingest" in sys.argv
    if "--realtime" in sys.argv:
        original_server.use_realtime = True
        original_server.agent_instance = None
    else:
        original_server.use_realtime = False
        original_server.agent_instance = None
    print(f"[Server] Ingest memories: {'on' if INGEST_MEMORIES else 'off'}")

    # Start Audio Threads
    threading.Thread(target=patched_receive_audio_from_robot, daemon=True).start()
    threading.Thread(target=patched_send_audio_to_robot, daemon=True).start()

    # Video Setup
    v_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    v_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    v_sock.bind((original_server.HOST, original_server.PORT_VIDEO_RX))
    v_sock.listen(1)
    
    print(f"[Video RX] Waiting for connection on {original_server.PORT_VIDEO_RX}...")
    conn, _ = v_sock.accept()
    payload_size = struct.calcsize(">L")

    # Frame stabilization & Recognition
    start_wait = time.time()
    first_frame = None
    
    while True:
        packed = original_server.recv_exact(conn, payload_size)
        if not packed: break
        msg_size = struct.unpack(">L", packed)[0]
        data = original_server.recv_exact(conn, msg_size)
        if not data: break
        
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None and (time.time() - start_wait) > RECOGNITION_FRAME_DELAY:
            first_frame = frame
            break
            
    if first_frame is not None:
        print("[Video] Recognizing user...")
        person = recognize_user_from_frame(first_frame)
        name = person.get("name") if person else None
        person_id = person.get("person_id") if person else None
        print(f"[User] {name or 'Unknown'}" + (f" (person_id={person_id})" if person_id else ""))

        if original_server.use_realtime:
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                print("[System] Starting Realtime Agent...")
                original_server.agent_instance = original_server.ServerRealtimeAgent(key, user_name=name, person_id=person_id)
                asyncio.create_task(original_server.agent_instance.run())

    # Main Loop
    rec_started = False
    try:
        while True:
            await asyncio.sleep(0.001)
            
            # Start recorder only after connection
            if not rec_started and original_server.use_realtime and original_server.agent_instance:
                if getattr(original_server.agent_instance, "session_connected", False):
                    recorder.start_recording()
                    rec_started = True
            
            recorder.check_and_save()

            # Video Loop
            packed = original_server.recv_exact(conn, payload_size)
            if not packed: break
            msg_size = struct.unpack(">L", packed)[0]
            data = original_server.recv_exact(conn, msg_size)
            if not data: break
            
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                recorder.add_video(frame)
                cv2.imshow("Server View", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        v_sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())