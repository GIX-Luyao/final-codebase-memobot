#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to sys.path immediately to ensure imports work
FILE_PATH = Path(__file__).resolve()
# memobot/robot/mac_master_v10.py -> memobot/robot -> memobot -> root
PROJECT_ROOT = FILE_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# Real-time spectral denoising (fan hum): use first 0.7s as noise profile, then strip
# with same logic as user_denoising.py (and test_audio_denoise.py).
try:
    from memobot.robot.user_denoising import (
        build_profile_from_noise,
        apply_profile_to_bytes,
        DEFAULT_NOISE_DURATION_SEC,
    )
    _user_denoising_available = True
except ImportError:
    build_profile_from_noise = apply_profile_to_bytes = None
    DEFAULT_NOISE_DURATION_SEC = 0.7
    _user_denoising_available = False

DENOISING_AVAILABLE = _user_denoising_available

# --- GEMINI LIVE (Google) REALTIME AGENT ---
try:
    from memobot.query_pipeline.gemini_client import ServerRealtimeAgent as GeminiServerAgent, set_code_command_socket
except ImportError:
    try:
        from query_pipeline.gemini_client import ServerRealtimeAgent as GeminiServerAgent, set_code_command_socket
    except ImportError:
        GeminiServerAgent = None
        set_code_command_socket = None

# Picovoice Porcupine for wake word (macOS ARM64 + Py3.11 friendly; no tflite)
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    pvporcupine = None
    PORCUPINE_AVAILABLE = False
    print("[Warning] pvporcupine not found. Install with: pip install pvporcupine")

# System microphone (external mic / headphone jack) for incoming audio
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None
    SOUNDDEVICE_AVAILABLE = False
    print("[Warning] sounddevice not found. Install with: pip install sounddevice")

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

# Porcupine: env PICOVOICE_ACCESS_KEY required (https://console.picovoice.ai/)
# Built-in keywords (use exact names): alexa, americano, blueberry, bumblebee, computer,
#   grapefruit, grasshopper, hey barista, hey google, hey siri, jarvis, ok google,
#   pico clock, picovoice, porcupine, terminator. Can list multiple to detect any.
PORCUPINE_KEYWORDS = ["jarvis"]

# Set in main() from --ingest: when True, run ingest pipeline after each 30s clip
INGEST_MEMORIES = False

# Code command channel: mock/robot client connects here to receive executed code (length-prefixed)
PORT_CODE_TX = 50009

# DSP: fan noise reduction (bandpass from v3) + spectral denoise (first 0.7s stream) + mic gain
MIC_INPUT_GAIN = 4.0  # Boost mic level (v3 uses 3.0); higher helps hear people further away
ROBOT_AUDIO_RATE = 16000  # Sample rate for real-time denoising

# GLOBAL: Keep track of the active code socket to send instant WAKE commands
_active_code_conn = None

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
                    kwargs={"duration_sec": None},
                    daemon=True,
                ).start()

    def stop_recording_and_finalize(self):
        """Stop recording and finalize the current clip (whatever length) and run ingest. Idempotent."""
        with self.lock:
            if not self.recording_started:
                return
            frames_to_save = list(self.video_buffer)
            user_bytes_to_save = bytes(self.user_audio_bytes)
            tts_chunks_to_save = list(self.tts_audio_chunks)
            clip_start_time = self.start_time
            self.video_buffer = []
            self.user_audio_bytes = bytearray()
            self.tts_audio_chunks = []
            self.recording_started = False
            self.start_time = None
        if frames_to_save or user_bytes_to_save or tts_chunks_to_save:
            # Compute actual duration from buffers (clip may be shorter than 30s)
            duration_from_video = len(frames_to_save) / self.fps if frames_to_save else 0
            duration_from_audio = len(user_bytes_to_save) / (self.sample_rate * 2) if user_bytes_to_save else 0
            duration_sec = max(duration_from_video, duration_from_audio, 0.1)
            if not frames_to_save:
                # Need at least one frame for video file; skip finalize if no video
                print("[Recorder] Session ended with no video frames; skipping clip save.")
                return
            threading.Thread(
                target=self._finalize_clip,
                args=(frames_to_save, user_bytes_to_save, tts_chunks_to_save, clip_start_time),
                kwargs={"duration_sec": duration_sec},
                daemon=True,
            ).start()
            print("[Recorder] Session ended; finalizing current clip and running ingest.")
        else:
            print("[Recorder] Session ended; no buffered clip to save.")

    def _finalize_clip(self, frames, user_bytes, tts_chunks, clip_start, duration_sec=None):
        if duration_sec is None:
            duration_sec = self.clip_duration
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        v_tmp = str(DATA_DIR / f"v_tmp_{ts}.mp4")
        a_tmp = str(DATA_DIR / f"a_tmp_{ts}.wav")
        final_path = str(DATA_DIR / f"clip_{ts}.mp4")

        try:
            total_samples = int(duration_sec * self.sample_rate)
            if total_samples <= 0:
                return
            
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
# Wake word: ignore repeated "jarvis" until session ends (goodbye); prevents multiple run() coroutines
_wake_word_triggered = False
_wake_word_triggered_lock = threading.Lock()

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

def _robot_audio_socket_keeper(sock):
    """Accept connections on robot audio port and read/discard data so the client can connect and send without blocking. Audio is not used; actual input comes from system mic."""
    while True:
        try:
            conn, addr = sock.accept()
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except OSError as e:
            if sock.fileno() == -1:
                return
            print(f"[Audio RX] Socket keeper: {e}")
        except Exception as e:
            print(f"[Audio RX] Socket keeper: {e}")


def patched_receive_audio_from_robot():
    """Uses the system microphone (external mic / headphone jack) for all incoming audio. Robot audio port is still opened so the client can connect (data is discarded). Porcupine (wake word) gets raw mic audio. When session is connected, applies bandpass + spectral denoise + gain and feeds recorder, VAD, TalkNet, API."""
    global _last_recognition_request_time, _user_audio_hold_until, _active_code_conn, _wake_word_triggered

    if not SOUNDDEVICE_AVAILABLE or sd is None:
        print("[Audio RX] ERROR: sounddevice is required for system mic. Install with: pip install sounddevice")
        return

    # Keep robot audio port open so client can connect (client code cannot be changed)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((original_server.HOST, original_server.PORT_AUDIO_RX))
    sock.listen(1)
    print(f"[Audio RX] Listening on {original_server.PORT_AUDIO_RX} (robot client may connect; audio from this port is not used).")
    threading.Thread(target=_robot_audio_socket_keeper, args=(sock,), daemon=True).start()

    # Actual audio input: system default microphone (e.g. headphone jack mic)
    MIC_BLOCKSIZE = 2048  # samples per read (2048 @ 16kHz ≈ 128ms), 4096 bytes
    print(f"[Audio RX] Using system microphone (default input device) at {ROBOT_AUDIO_RATE} Hz. Blocksize={MIC_BLOCKSIZE} samples.")

    # Fan noise reduction: bandpass 300Hz–4kHz (removes NAO fan rumble + hiss). Uses v3's RealtimeFilter.
    noise_filter = original_server.RealtimeFilter(
        original_server.FILTER_LOW_CUT,
        original_server.FILTER_HIGH_CUT,
        original_server.ROBOT_RATE,
    )
    if getattr(noise_filter, "enabled", False):
        print("[Audio RX] Fan noise filter enabled (bandpass 300–4000 Hz).")
    print(f"[Audio RX] Mic input gain: {MIC_INPUT_GAIN}x (for distant speech).")

    # Streaming noise profile: first 0.7s of session audio → build profile (user_denoising), then denoise each chunk.
    NOISE_DURATION_SEC = DEFAULT_NOISE_DURATION_SEC
    noise_profile_bytes_needed = int(ROBOT_AUDIO_RATE * NOISE_DURATION_SEC * 2)  # mono int16
    noise_buffer = bytearray()
    denoising_profile = None  # DenoisingProfile set once we have 0.7s
    if DENOISING_AVAILABLE and original_server.ROBOT_RATE == ROBOT_AUDIO_RATE:
        print(f"[Audio RX] Spectral denoising: will use first {NOISE_DURATION_SEC}s of stream as fan noise profile (user_denoising).")

    vad_model = None
    if SILERO_VAD_AVAILABLE:
        try:
            vad_model = load_silero_vad()
            print("[VAD] Silero VAD loaded.")
        except Exception as e:
            print(f"[VAD] Failed to load Silero VAD: {e}")

    porcupine = None
    if PORCUPINE_AVAILABLE and pvporcupine is not None:
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if access_key:
            try:
                porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=PORCUPINE_KEYWORDS,
                )
                print(f"[Wake Word] Porcupine loaded. Listening for: {PORCUPINE_KEYWORDS}")
            except Exception as e:
                print(f"[Wake Word] Failed to load Porcupine: {e}")
                porcupine = None
        else:
            print("[Wake Word] PICOVOICE_ACCESS_KEY not set. Get a free key at https://console.picovoice.ai/")

    vad_buffer = bytearray()
    porcupine_buffer = bytearray()
    porcupine_frame_bytes = (porcupine.frame_length * 2) if porcupine is not None else 0
    in_speech_prev = False

    # Try to open microphone stream with fallback logic for Linux/Ubuntu
    mic_stream = None

    def _try_open_stream(device_idx=None):
        s = sd.InputStream(
            samplerate=ROBOT_AUDIO_RATE,
            channels=1,
            dtype="int16",
            blocksize=MIC_BLOCKSIZE,
            device=device_idx,
        )
        s.start()
        return s

    # 1. First Priority: Default Device (Standard behavior, preferred for Mac)
    try:
        mic_stream = _try_open_stream(None)
        print("[Audio RX] Started system microphone (default device).")
    except Exception as e:
        print(f"[Audio RX] Default mic failed: {e}. Searching for fallback devices (Linux compatibility)...")
        # 2. Linux/Ubuntu Fallback: Search for valid inputs
        try:
            devices = sd.query_devices()
            candidates = []
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    name = d.get('name', '').lower()
                    score = 0
                    if 'usb' in name: score += 10
                    if 'default' in name: score += 5
                    if 'pulse' in name: score += 4
                    if 'sysdefault' in name: score += 3
                    candidates.append((score, i, d.get('name', 'Unknown')))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, idx, name in candidates:
                try:
                    print(f"[Audio RX] Fallback: Trying device {idx} ('{name}')...")
                    mic_stream = _try_open_stream(idx)
                    print(f"[Audio RX] Success! Using device {idx}: {name}")
                    break
                except Exception as ex:
                    print(f"[Audio RX] Device {idx} failed: {ex}")
        except Exception as e2:
             print(f"[Audio RX] Device enumeration failed: {e2}")

    if mic_stream is None:
        print("[Audio RX] CRITICAL: Failed to open any microphone input. Check system settings.")
        return

    try:
        while True:
            try:
                indata, _ = mic_stream.read(MIC_BLOCKSIZE)
            except Exception as e:
                print(f"[Audio RX] Mic read error: {e}")
                break
            if indata is None or indata.size == 0:
                continue
            # (samples, channels) -> flatten and to bytes
            data = indata.reshape(-1).tobytes()
            raw_data = data  # Porcupine gets raw audio only (no denoising)

            # Check if session is active — only then apply denoising and feed recorder/VAD/API
            session_active = (
                original_server.use_realtime and
                original_server.agent_instance and
                getattr(original_server.agent_instance, "session_connected", False)
            )

            if session_active:
                # Connected: apply full pipeline (bandpass → spectral denoise → gain), then feed downstream
                data = noise_filter.process(data)
                # Build noise profile from first 0.7s of stream (user_denoising), then denoise each chunk the same way
                if DENOISING_AVAILABLE and original_server.ROBOT_RATE == ROBOT_AUDIO_RATE:
                    if denoising_profile is None:
                        noise_buffer.extend(data)
                        if len(noise_buffer) >= noise_profile_bytes_needed:
                            raw_noise = bytes(noise_buffer[:noise_profile_bytes_needed])
                            noise_part = np.frombuffer(raw_noise, dtype=np.int16)
                            noise_buffer = bytearray()
                            denoising_profile = build_profile_from_noise(
                                noise_part, ROBOT_AUDIO_RATE,
                                prop_decrease=0.97,
                                n_fft=8192,
                            )
                            print("[Audio RX] Fan noise profile ready (0.7s). Spectral denoising enabled (user_denoising).")
                    if denoising_profile is not None:
                        try:
                            data = apply_profile_to_bytes(
                                denoising_profile,
                                data,
                                channels=1,
                            )
                        except Exception:
                            pass
                data = original_server.apply_gain(data, MIC_INPUT_GAIN)

                recorder.add_audio(data)
                if TALKNET_AVAILABLE:
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    _audio_ring_buffer.extend(audio_np)

                # Feed processed audio to realtime agent
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

                # VAD on processed audio (for face recognition on speech start)
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

            else:
                # Idle: feed only raw audio to Porcupine for wake word (no denoising)
                if porcupine is not None and porcupine_frame_bytes > 0:
                    porcupine_buffer.extend(raw_data)
                    while len(porcupine_buffer) >= porcupine_frame_bytes:
                        chunk = bytes(porcupine_buffer[:porcupine_frame_bytes])
                        del porcupine_buffer[:porcupine_frame_bytes]
                        pcm = np.frombuffer(chunk, dtype=np.int16).tolist()
                        try:
                            keyword_index = porcupine.process(pcm)
                            if keyword_index >= 0:
                                # Ignore repeated wake word until session has ended (goodbye)
                                with _wake_word_triggered_lock:
                                    if _wake_word_triggered:
                                        continue  # Session already started; ignore until goodbye
                                    _wake_word_triggered = True
                                kw = PORCUPINE_KEYWORDS[keyword_index] if keyword_index < len(PORCUPINE_KEYWORDS) else "?"
                                print(f"[Wake Word] 🌟 Detected '{kw}'. Starting Gemini Live...")
                                
                                # --- INJECT COMMAND WAKE ---
                                if _active_code_conn is not None:
                                    try:
                                        wake_msg = b"WAKE"
                                        _active_code_conn.sendall(struct.pack(">L", len(wake_msg)) + wake_msg)
                                        print("[Wake Word] Sent instant WAKE command to client code channel.")
                                    except Exception as e:
                                        print(f"[Wake Word] Failed to send WAKE command: {e}")
                                # ---------------------------

                                if original_server.loop_instance and original_server.agent_instance:
                                    original_server.agent_instance.ingest_robot_audio(chunk)
                                    original_server.loop_instance.call_soon_threadsafe(
                                        lambda: asyncio.create_task(original_server.agent_instance.run())
                                    )
                                else:
                                    with _wake_word_triggered_lock:
                                        _wake_word_triggered = False  # Allow retry if agent wasn't ready
                                    print("[Wake Word] ⚠ Gemini Live not ready. Run with --realtime and set GOOGLE_API_KEY.")
                        except Exception as e:
                            print(f"[Wake Word] process error: {e}")
                    if len(porcupine_buffer) > porcupine_frame_bytes * 4:
                        porcupine_buffer = porcupine_buffer[-porcupine_frame_bytes:]
    except Exception as e:
        print(f"[Audio RX] Error: {e}")
    finally:
        if porcupine is not None:
            try:
                porcupine.delete()
            except Exception:
                pass
        try:
            mic_stream.stop()
            mic_stream.close()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

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
    global _active_code_conn
    while True:
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((original_server.HOST, PORT_CODE_TX))
            server_sock.listen(1)
            print(f"[Code TX] Listening on {PORT_CODE_TX} (executed code channel)...")
            conn, addr = server_sock.accept()
            print(f"[Code TX] Client connected: {addr}")
            
            _active_code_conn = conn
            if set_code_command_socket is not None:
                set_code_command_socket(conn)
            
            try:
                while True:
                    if conn.recv(4096) == b"":
                        break
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                _active_code_conn = None
                if set_code_command_socket is not None:
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

    # Create Gemini Live agent *before* blocking on video, so wake word can start session
    # even when only the audio client has connected.
    if original_server.use_realtime:
        key = os.environ.get("GOOGLE_API_KEY")
        if key and GeminiServerAgent is not None:
            print("[System] Initializing Gemini Live Agent...")
            print("[System] ⏳ Say the wake word (e.g. 'jarvis') to start session. Set PICOVOICE_ACCESS_KEY.")
            original_server.agent_instance = GeminiServerAgent(
                api_key=key,
                user_name=None,
                person_id=None,
                audio_tx_queue=original_server.audio_tx_queue,
                loop=original_server.loop_instance,
            )
            # Do NOT start run() yet. Waiting for wake word.
        elif not key:
            print("[System] GOOGLE_API_KEY not set; realtime mode disabled.")
        elif GeminiServerAgent is None:
            print("[System] Gemini client not available; realtime mode disabled.")

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
    global _last_speaker_sent_to_api, _latest_frame, _wake_word_triggered
    rec_started = False
    session_was_connected = False
    try:
        while True:
            await asyncio.sleep(0.001)

            session_connected = (
                original_server.use_realtime
                and original_server.agent_instance
                and getattr(original_server.agent_instance, "session_connected", False)
            )

            # Start recorder only when Gemini is connected
            if not rec_started and session_connected:
                recorder.start_recording()
                rec_started = True
                session_was_connected = True

            # When connection drops (goodbye): finalize clip, stop recording, allow wake word again
            if rec_started and session_was_connected and not session_connected:
                recorder.stop_recording_and_finalize()
                rec_started = False
                session_was_connected = False
                with _wake_word_triggered_lock:
                    _wake_word_triggered = False  # Say "Jarvis" again to start a new session

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
                # Only record video when Gemini is connected (no recording when disconnected)
                if session_connected:
                    recorder.add_video(frame)
                cv2.imshow("Server View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        v_sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())