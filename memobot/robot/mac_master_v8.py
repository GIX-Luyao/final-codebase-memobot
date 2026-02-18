#!/usr/bin/env python3
import sys
import time
import queue
import cv2
import struct
import socket
import threading
import asyncio
import uuid
import numpy as np
import wave
import subprocess
import collections
from pathlib import Path
from datetime import datetime

# Silero VAD for voice-activity-driven recognition
try:
    import torch
    from silero_vad import load_silero_vad
    SILERO_VAD_AVAILABLE = True
except ImportError:
    torch = None
    SILERO_VAD_AVAILABLE = False

# TalkNet for active speaker detection
try:
    # Need to add talknet directory to path
    import sys
    talknet_dir = str(Path(__file__).resolve().parent.parent / "ingest_pipeline" / "talknet")
    if talknet_dir not in sys.path:
        sys.path.insert(0, talknet_dir)
    from realtime_talknet import RealtimeTalkNet
    TALKNET_MODEL_PATH = Path(talknet_dir) / "pretrain_TalkSet.model"
    # Select device- check if MPS or CUDA available
    device = 'cpu'
    if torch and torch.cuda.is_available():
        device = 'cuda'
    # MPS support for Mac requires pytorch specific version, stick to catch-all
    elif torch and torch.backends.mps.is_available():
        device = 'mps'
        
    talknet_engine = RealtimeTalkNet(str(TALKNET_MODEL_PATH), device=device)
    TALKNET_AVAILABLE = True
    print(f"[TalkNet] Loaded successfully on {device}")
except ImportError as e:
    print(f"[TalkNet] Not available: {e}")
    TALKNET_AVAILABLE = False
except Exception as e:
    print(f"[TalkNet] Failed to load: {e}")
    TALKNET_AVAILABLE = False

# --- IMPORT ORIGINAL SERVER LOGIC (sockets, ports, recv_exact, audio_tx_queue) ---
import memobot.robot.mac_master_v3 as original_server

# --- GEMINI LIVE (Google) REALTIME AGENT ---
try:
    from memobot.query_pipeline.gemini_client import ServerRealtimeAgent as GeminiServerAgent, set_code_command_socket
except ImportError:
    try:
        from query_pipeline.gemini_client import ServerRealtimeAgent as GeminiServerAgent, set_code_command_socket
    except ImportError:
        GeminiServerAgent = None
        set_code_command_socket = None

# --- CONFIGURATION ---
ROBOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROBOT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "ingest_pipeline" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RECOGNIZED_USER_FRAMES_DIR = ROBOT_DIR / "recognized_user_frames"
RECOGNIZED_USER_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
TALKNET_CROPS_DIR = ROBOT_DIR / "talknet_crops"  # Crops for debugging only; do NOT put in face_database
TALKNET_CROPS_DIR.mkdir(parents=True, exist_ok=True)
FACE_DATABASE_DIR = REPO_ROOT / "face_database"

MAX_RECOGNITION_DISTANCE = 0.5   # Below this: confident same person
# Above this: best match is too far → treat as new user and register. Higher = stricter (fewer false new entries).
MAX_DISTANCE_TO_ACCEPT_AS_SAME_PERSON = 0.6
NETWORK_LATENCY_DELAY = 0.25
# Minimum face size (width/height in px) to accept when saving to face_database; avoids saving non-face or tiny detections.
MIN_FACE_SIZE_FOR_ENROLL = 60

# VAD-driven recognition
VAD_CHUNK_SAMPLES = 512   # 32ms at 16kHz (Silero recommends 30ms+)
VAD_SPEECH_THRESHOLD = 0.5
VAD_RECOGNITION_COOLDOWN = 2.0  # seconds between recognition requests on speech start
# Hold user audio from API for this long after speech start so speaker can be updated before the turn is processed
USER_AUDIO_HOLD_AFTER_SPEECH_START = 0.5

# Set in main() from --ingest: when True, run ingest pipeline after each 30s clip
INGEST_MEMORIES = False

# Code command channel: mock/robot client connects here to receive executed code (length-prefixed)
PORT_CODE_TX = 50009

# DSP: fan noise reduction (bandpass from v3) + mic sensitivity for distant speech
MIC_INPUT_GAIN = 4.0  # Boost mic level (v3 uses 3.0); higher helps hear people further away

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "query_pipeline") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "query_pipeline"))

from utils.database import init_database, add_person, get_person_by_face_id, delete_person_by_face_id
from utils.main import update_face_embedding_cache

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

# --- VAD-driven recognition shared state ---
_latest_frame_lock = threading.Lock()
_latest_frame = None  # numpy array copy of most recent video frame
# TalkNet: Rolling buffers (1 second window) - 25 FPS video, 16kHz audio
_video_ring_buffer = collections.deque(maxlen=25)
_audio_ring_buffer = collections.deque(maxlen=16000)
_recognition_requested = threading.Event()
# Serialize face recognition: TensorFlow/Metal crashes if two recognitions run concurrently
_recognition_lock = threading.Lock()
_current_speaker_lock = threading.Lock()
_current_speaker = None  # {"person_id": ..., "name": ...} or None
_last_speaker_sent_to_api = None  # name last sent via session.update (or None)
_last_recognition_request_time = 0.0
_last_recognition_request_lock = threading.Lock()
# User audio hold: buffer mic for a short window after speech start so recognition can update speaker before API hears the turn
_user_audio_hold_until = 0.0
_user_audio_buffer = bytearray()
_user_audio_buffer_lock = threading.Lock()

def _int16_to_float32(audio_int16):
    """Convert int16 audio to float32 for Silero VAD (-1..1)."""
    audio = np.frombuffer(audio_int16, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0

# --- RECOGNITION ---
def register_unknown_user(frame):
    """Assign a new Person ID, add to persons.db, then save face image so they are recognized next time.
    persons.db is always updated whenever a new face identity is added.
    Does nothing and returns None if the frame contains no detectable face of sufficient size (avoids saving empty/bad crops)."""
    faces = detect_faces_fast(frame)
    if not faces:
        print("[Recognition] No face detected in frame; not saving to face_database.")
        return None
    valid_faces = [f for f in faces if f.get("w", 0) >= MIN_FACE_SIZE_FOR_ENROLL and f.get("h", 0) >= MIN_FACE_SIZE_FOR_ENROLL]
    if not valid_faces:
        print("[Recognition] No face large enough detected in frame; not saving to face_database.")
        return None
    face_id = str(uuid.uuid4())
    name = f"Guest_{face_id[:8]}"
    init_database(verbose=False)
    try:
        add_person(face_id=face_id, name=name)
    except Exception as e:
        print(f"[Recognition] Failed to add person to persons.db: {e}")
        return None
    FACE_DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = FACE_DATABASE_DIR / f"{face_id}.png"
    try:
        cv2.imwrite(str(dest_path), frame)
    except Exception as e:
        print(f"[Recognition] Failed to save face image: {e}")
        delete_person_by_face_id(face_id)
        return None
    if not update_face_embedding_cache(face_id, dest_path):
        print(f"[Recognition] Warning: could not update face embedding cache for {face_id}")
    person = get_person_by_face_id(face_id)
    if person:
        print(f"[Recognition] Registered new user: person_id={person['person_id']}, name={name}")
    return person


def detect_faces_fast(frame):
    """
    Fast face detection using OpenCV Haar Cascade.
    Returns list of dicts: [{'x':x, 'y':y, 'w':w, 'h':h}, ...]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load cascade (try standard paths)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("[Detection] Error: Could not load face cascade")
        return []
        
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    for (x, y, w, h) in faces:
        results.append({'x': x, 'y': y, 'w': w, 'h': h})
    
    return results


def recognize_user_from_frame(frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = DATA_DIR / f"first_frame_{ts}.png"
    cv2.imwrite(str(tmp_path), frame)
    try:
        from recognize_user import recognize_user
        result = recognize_user(image_path=str(tmp_path), verbose=False)
        tmp_path.unlink(missing_ok=True)
        if result is None:
            return None
        # Face detected but not in database: caller may register (only when confident it's new)
        if isinstance(result, dict) and result.get("unknown"):
            return {"unknown": True}
        person = result
        name = person.get("name", "Unknown").replace(" ", "_")
        dist = person.get("distance", 1.0)
        if dist > MAX_RECOGNITION_DISTANCE:
            if dist > MAX_DISTANCE_TO_ACCEPT_AS_SAME_PERSON:
                # Best match is too far: likely a different person (new user). Register.
                print(f"[Recognition] Distance {dist:.4f} > {MAX_DISTANCE_TO_ACCEPT_AS_SAME_PERSON}; treating as new user.")
                return {"unknown": True}
            # In between: uncertain (same person with bad crop). Do NOT register and do NOT update current speaker.
            print(f"[Recognition] Low confidence ({dist:.4f}). Ignoring this detection (keeping current speaker).")
            cv2.imwrite(str(RECOGNIZED_USER_FRAMES_DIR / f"{ts}_{name}_low_conf.png"), frame)
            return {"low_confidence": True, "distance": dist}
        cv2.imwrite(str(RECOGNIZED_USER_FRAMES_DIR / f"{ts}_{name}.png"), frame)
        return person
    except Exception as e:
        print(f"[Recognition] Error: {e}")
        tmp_path.unlink(missing_ok=True)
        return None

# --- PATCHED NETWORKING ---

def patched_receive_audio_from_robot():
    """Reads audio from socket in a loop; applies fan noise reduction (bandpass) + mic gain; feeds recorder, VAD, API."""
    global _last_recognition_request_time, _user_audio_hold_until

    print(f"[Audio RX] Listening on {original_server.PORT_AUDIO_RX}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((original_server.HOST, original_server.PORT_AUDIO_RX))
    sock.listen(1)

    # Fan noise reduction: bandpass 300Hz–4kHz (removes NAO fan rumble + hiss). Uses v3's RealtimeFilter.
    noise_filter = original_server.RealtimeFilter(
        original_server.FILTER_LOW_CUT,
        original_server.FILTER_HIGH_CUT,
        original_server.ROBOT_RATE,
    )
    if getattr(noise_filter, "enabled", False):
        print("[Audio RX] Fan noise filter enabled (bandpass 300–4000 Hz).")
    print(f"[Audio RX] Mic input gain: {MIC_INPUT_GAIN}x (for distant speech).")

    vad_model = None
    if SILERO_VAD_AVAILABLE:
        try:
            vad_model = load_silero_vad()
            print("[VAD] Silero VAD loaded.")
        except Exception as e:
            print(f"[VAD] Failed to load Silero VAD: {e}")

    vad_buffer = bytearray()
    in_speech_prev = False

    conn, _ = sock.accept()
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break

            # 1. Fan noise reduction (bandpass)
            data = noise_filter.process(data)
            # 2. Mic sensitivity boost for distant speakers
            data = original_server.apply_gain(data, MIC_INPUT_GAIN)

            recorder.add_audio(data)
            
            # TalkNet: Update audio ring buffer
            if TALKNET_AVAILABLE:
                audio_np = np.frombuffer(data, dtype=np.int16)
                _audio_ring_buffer.extend(audio_np)

            # Feed realtime agent (user mic -> API). Hold user audio briefly after speech start so speaker can be updated first.
            if original_server.use_realtime and original_server.agent_instance:
                if not original_server.agent_instance.is_robot_speaking():
                    with _user_audio_buffer_lock:
                        now = time.time()
                        if now < _user_audio_hold_until:
                            _user_audio_buffer.extend(data)
                        else:
                            if _user_audio_buffer:
                                flush = bytes(_user_audio_buffer)
                                _user_audio_buffer.clear()
                                original_server.agent_instance.ingest_robot_audio(flush)
                            original_server.agent_instance.ingest_robot_audio(data)

            # VAD: run on user mic only; on speech start request face recognition
            if vad_model is not None:
                vad_buffer.extend(data)
                chunk_bytes = VAD_CHUNK_SAMPLES * 2  # 16-bit
                while len(vad_buffer) >= chunk_bytes:
                    chunk = bytes(vad_buffer[:chunk_bytes])
                    del vad_buffer[:chunk_bytes]
                    audio_f32 = _int16_to_float32(chunk)
                    try:
                        confidence = vad_model(torch.from_numpy(audio_f32), 16000).item()
                    except Exception:
                        confidence = 0.0
                    in_speech = confidence >= VAD_SPEECH_THRESHOLD
                    if not in_speech_prev and in_speech:
                        with _last_recognition_request_lock:
                            if (time.time() - _last_recognition_request_time) >= VAD_RECOGNITION_COOLDOWN:
                                _recognition_requested.set()
                                _last_recognition_request_time = time.time()
                                with _user_audio_buffer_lock:
                                    _user_audio_hold_until = time.time() + USER_AUDIO_HOLD_AFTER_SPEECH_START
                    in_speech_prev = in_speech
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


def thread_code_command_channel():
    """Listen on PORT_CODE_TX (50009). When mock/robot client connects, register socket so executeCode can send code."""
    if set_code_command_socket is None:
        return
    while True:
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((original_server.HOST, PORT_CODE_TX))
            server_sock.listen(1)
            print(f"[Code TX] Listening on {PORT_CODE_TX} (executed code channel)...")
            conn, addr = server_sock.accept()
            print(f"[Code TX] Client connected: {addr}")
            set_code_command_socket(conn)
            try:
                while True:
                    if conn.recv(4096) == b"":
                        break
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                set_code_command_socket(None)
                conn.close()
            server_sock.close()
        except Exception as e:
            print(f"[Code TX] Error: {e}")
            try:
                server_sock.close()
            except Exception:
                pass


# --- MAIN ---

async def main():
    original_server.loop_instance = asyncio.get_running_loop()
    
    # Mode Setup
    import os
    global INGEST_MEMORIES, _current_speaker
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
    # Code command channel: mock/robot client connects to receive executed code
    threading.Thread(target=thread_code_command_channel, daemon=True).start()

    # Video Setup
    v_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    v_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    v_sock.bind((original_server.HOST, original_server.PORT_VIDEO_RX))
    v_sock.listen(1)
    
    print(f"[Video RX] Waiting for connection on {original_server.PORT_VIDEO_RX}...")
    conn, _ = v_sock.accept()
    payload_size = struct.calcsize(">L")

    # Start realtime agent (Gemini Live API); user detected on first speech via VAD
    if original_server.use_realtime:
        key = os.environ.get("GOOGLE_API_KEY")
        if key and GeminiServerAgent is not None:
            print("[System] Starting Gemini Live Agent (user will be detected on first speech)...")
            original_server.agent_instance = GeminiServerAgent(
                api_key=key,
                user_name=None,
                person_id=None,
                audio_tx_queue=original_server.audio_tx_queue,
                loop=original_server.loop_instance,
            )
            asyncio.create_task(original_server.agent_instance.run())
        elif not key:
            print("[System] GOOGLE_API_KEY not set; realtime mode disabled.")
        elif GeminiServerAgent is None:
            print("[System] Gemini client not available; realtime mode disabled.")

    def _run_recognition_on_frame(frame_copy):
        """Run face recognition pipeline: VAD -> TalkNet -> DeepFace."""
        global _current_speaker
        t_start = time.time()
        
        # 1. TalkNet Active Speaker Detection
        active_speaker_crop = None
        used_optimized_pipeline = False
        
        if TALKNET_AVAILABLE:
            with _latest_frame_lock:
                video_history = list(_video_ring_buffer)
                audio_history = np.array(_audio_ring_buffer)
            
            # Only run if we have enough history (~0.8s+)
            if len(video_history) >= 20 and len(audio_history) > 12000:
                try:
                    faces = detect_faces_fast(frame_copy)
                    if faces:
                        # Convert to [x1, y1, x2, y2]
                        formatted_faces = []
                        for f in faces:
                            formatted_faces.append([f['x'], f['y'], f['x']+f['w'], f['y']+f['h']])
                        
                        active_bbox = talknet_engine.predict_active_speaker(
                            video_history,
                            audio_history,
                            formatted_faces
                        )
                        
                        if active_bbox:
                            x1, y1, x2, y2 = active_bbox
                            # Add some margin for DeepFace/Alignment
                            h, w, _ = frame_copy.shape
                            margin_x = int((x2-x1) * 0.2)
                            margin_y = int((y2-y1) * 0.2)
                            cx1 = max(0, x1 - margin_x)
                            cy1 = max(0, y1 - margin_y)
                            cx2 = min(w, x2 + margin_x)
                            cy2 = min(h, y2 + margin_y)
                            
                            active_speaker_crop = frame_copy[cy1:cy2, cx1:cx2]
                            
                            # Only save crop when we have an active speaker (red box)
                            if active_speaker_crop.size > 0:
                                try:
                                    ts_crop = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    crop_output_path = TALKNET_CROPS_DIR / f"talknet_crop_{ts_crop}.png"
                                    cv2.imwrite(str(crop_output_path), active_speaker_crop)
                                    print(f"[TalkNet] Saved crop to {crop_output_path}")
                                except Exception as e:
                                    print(f"[TalkNet] Failed to save crop: {e}")
                                print(f"[TalkNet] Active speaker found at {active_bbox}. Cropped for ID.")
                                used_optimized_pipeline = True
                        else:
                            # No active speaker (no red box): do not crop, do not run recognition this turn
                            print("[TalkNet] No active speaker detected; skipping crop and recognition.")
                            return
                except Exception as e:
                    print(f"[TalkNet] Error in pipeline: {e}")
        
        t_talknet = time.time()
        
        # 2. DeepFace identification
        # If we have a valid crop, recognize that. Else fallback to full frame.
        target_image = frame_copy
        if active_speaker_crop is not None and active_speaker_crop.size > 0:
            target_image = active_speaker_crop
        elif TALKNET_AVAILABLE:
            # If TalkNet was available but failed to produce a valid crop, 
            # we might want to be careful. But fallback is full frame.
            pass
        
        # Note: recognize_user_from_frame expects a full frame usually regarding logic, 
        # but recognize_user library works on image files. 
        # If we pass a crop, it should work better as there is only one face.
        
        result = recognize_user_from_frame(target_image)
        
        t_end = time.time()
        total_time = t_end - t_start
        talknet_time = t_talknet - t_start
        deepface_time = t_end - t_talknet
        print(f"[Latency] Total: {total_time:.3f}s (TalkNet: {talknet_time:.3f}s, Recognition: {deepface_time:.3f}s)")
        
        if result is None:
            # If we used crop and failed, maybe try full frame as backup? 
            # For now, just clear speaker.
            if used_optimized_pipeline:
                 pass # Keep existing speaker or clear?
            
            # If nobody recognized in full frame either (or crop failed)
            # Only clear if we were doing a full scan or if we are sure nobody is there
            # For now, simplistic approach:
            # with _current_speaker_lock:
            #    _current_speaker = None
            return

        if isinstance(result, dict) and result.get("low_confidence"):
            # Low-confidence match: do NOT update current speaker or notify the Live API.
            return

        if isinstance(result, dict) and result.get("unknown"):
            # Only register when recognizer found no match in DB (truly new person). Low-confidence
            # matches are handled above by returning early.
            person = register_unknown_user(target_image)
            with _current_speaker_lock:
                _current_speaker = (
                    {"person_id": person["person_id"], "name": person["name"], "is_new_user": True}
                    if person is not None
                    else None
                )
            return
            
        person = result
        with _current_speaker_lock:
            _current_speaker = {"person_id": person.get("person_id"), "name": person.get("name")}

    # Main Loop
    global _last_speaker_sent_to_api, _latest_frame
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

            # VAD-driven recognition: when speech start was detected, run face recognition on latest frame.
            # Only one recognition at a time (TensorFlow/Metal crashes on concurrent model execution).
            if _recognition_requested.is_set():
                _recognition_requested.clear()
                with _latest_frame_lock:
                    frame_copy = _latest_frame.copy() if _latest_frame is not None else None
                if frame_copy is not None and _recognition_lock.acquire(blocking=False):
                    def _run_then_release():
                        try:
                            _run_recognition_on_frame(frame_copy)
                        finally:
                            _recognition_lock.release()
                    threading.Thread(target=_run_then_release, daemon=True).start()

            # session.update: as soon as facial recognition identifies/changes user, update Realtime API
            with _current_speaker_lock:
                speaker = _current_speaker
            name_for_api = speaker.get("name") if speaker else None
            if name_for_api != _last_speaker_sent_to_api:
                person_id = speaker.get("person_id") if speaker else None
                is_new_user = speaker.get("is_new_user", False) if speaker else False
                print(f"[Person Record]: person_id: {person_id}, name: {name_for_api or 'Unknown'}" + (" (new user)" if is_new_user else ""))
                if original_server.use_realtime and original_server.agent_instance and getattr(original_server.agent_instance, "session_connected", False):
                    print(f"[Live API] Sending speaker identity to Gemini: name={name_for_api!r}, person_id={person_id!r}, is_new_user={is_new_user}")
                    await original_server.agent_instance.update_speaker_identity(name_for_api, person_id=person_id, is_new_user=is_new_user)
                _last_speaker_sent_to_api = name_for_api
                # Clear is_new_user after notifying API so we don't repeat "new user" on next loop
                if is_new_user and speaker:
                    with _current_speaker_lock:
                        if _current_speaker and _current_speaker.get("name") == name_for_api:
                            _current_speaker = {**_current_speaker, "is_new_user": False}

            # Video Loop
            packed = original_server.recv_exact(conn, payload_size)
            if not packed:
                break
            msg_size = struct.unpack(">L", packed)[0]
            data = original_server.recv_exact(conn, msg_size)
            if not data:
                break

            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                with _latest_frame_lock:
                    _latest_frame = frame.copy()
                    if TALKNET_AVAILABLE:
                        _video_ring_buffer.append(frame.copy())
                recorder.add_video(frame)
                cv2.imshow("Server View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        v_sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())