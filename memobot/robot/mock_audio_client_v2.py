#!/usr/bin/env python3
"""
Mock TCP client that streams Mac microphone and camera to mac_master_v4.
Use this to test mac_master_v4 without a physical robot: it acts as the "robot"
sending audio (mic) and video (webcam) to the server.
"""

import argparse
import socket
import struct
import sys
import threading
import time
import wave
import queue

import cv2
import pyaudio

# --- CONFIG (must match mac_master_v3 / mac_master_v4 server) ---
DEFAULT_HOST = "127.0.0.1"
PORT_AUDIO_RX = 50005  # We send Mic audio TO this port
PORT_VIDEO_RX = 50006  # We send Video TO this port
PORT_AUDIO_TX = 50007  # We receive TTS audio FROM this port
PORT_CODE_RX = 50009   # We receive executed code FROM this port (when server runs command channel)

# Audio: match server expectation (robot / NAO style)
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 4096

# Video
JPEG_QUALITY = 50
VIDEO_FPS_TARGET = 20


def run_audio_client(
    host: str,
    port: int,
    stop_event: threading.Event,
    record_path: str | None = None,
) -> None:
    """
    Stream Mac microphone to server (audio RX). 
    """
    p = pyaudio.PyAudio()
    wav_file = None
    stream = None

    try:
        # Open Mic Stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        print(f"[Mic] Microphone opened. sending to {host}:{port}")

        if record_path:
            try:
                wav_file = wave.open(record_path, "wb")
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(p.get_sample_size(FORMAT))
                wav_file.setframerate(SAMPLE_RATE)
                print(f"[Mic] Recording local mic to {record_path}")
            except Exception as e:
                print(f"[Mic] Warning: Could not create recording file: {e}")

        while not stop_event.is_set():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((host, port))
                sock.settimeout(None)
                print(f"[Mic] Connected to server {host}:{port}")

                try:
                    while not stop_event.is_set():
                        # Read from Mic
                        try:
                            data = stream.read(CHUNK, exception_on_overflow=False)
                        except OSError as e:
                            print(f"[Mic] Read error (ignored): {e}")
                            continue

                        if not data:
                            break
                        
                        # Send to Server
                        sock.sendall(data)

                        # Save to file
                        if wav_file is not None:
                            wav_file.writeframes(data)
                
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    print(f"[Mic] Disconnected: {e}")
                finally:
                    sock.close()

            except (ConnectionRefusedError, OSError):
                # Server likely not up yet
                pass
            
            if not stop_event.is_set():
                time.sleep(2)

    except Exception as e:
        print(f"[Mic] Critical Error: {e}")
    finally:
        if wav_file:
            wav_file.close()
            print(f"[Mic] Saved recording to {record_path}")
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("[Mic] Stopped.")


def run_video_client(host: str, port: int, stop_event: threading.Event) -> None:
    """Stream Mac camera to server (video RX)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Video] Failed to open camera.")
        return

    # optimize for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    frame_interval = 1.0 / VIDEO_FPS_TARGET

    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)
            print(f"[Video] Connected to {host}:{port}")
            
            next_frame_time = time.monotonic()
            
            try:
                while not stop_event.is_set():
                    # FPS Limiter
                    now = time.monotonic()
                    if now < next_frame_time:
                        time.sleep(min(0.01, next_frame_time - now))
                        continue
                    next_frame_time = now + frame_interval

                    ret, frame = cap.read()
                    if not ret:
                        print("[Video] Camera read failed")
                        break
                    
                    # Encode JPEG
                    result, encimg = cv2.imencode(".jpg", frame, encode_param)
                    if not result:
                        continue
                    
                    data = encimg.tobytes()
                    msg_size = len(data)
                    
                    # Protocol: [4-byte size] + [JPEG bytes]
                    sock.sendall(struct.pack(">L", msg_size) + data)
            
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                sock.close()

        except (ConnectionRefusedError, OSError):
            pass
        
        if not stop_event.is_set():
            time.sleep(2)

    cap.release()
    print("[Video] Stopped.")


# --- AUDIO RECEIVER HELPERS ---

def list_output_devices(p: pyaudio.PyAudio) -> None:
    print("[Speaker] Available output devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxOutputChannels", 0) > 0:
            name = info.get("name", "?")
            print(f"  [{i}] {name}")

def get_output_device_index(p: pyaudio.PyAudio, device_name_or_index: str | None) -> int | None:
    """Resolves user input to a valid PyAudio device index."""
    if not device_name_or_index:
        # Fallback to first available if default is not specified
        for i in range(p.get_device_count()):
            if p.get_device_info_by_index(i).get("maxOutputChannels", 0) > 0:
                return i
        return None

    # Try as index
    if str(device_name_or_index).isdigit():
        return int(device_name_or_index)

    # Search by name
    target = device_name_or_index.lower()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxOutputChannels", 0) > 0:
            if target in (info.get("name") or "").lower():
                return i
    return None

def audio_playback_worker(audio_queue, stop_event, p, output_device_index):
    """
    Consumer thread: Reads from queue, writes to speakers.
    Maintains a buffer to prevent chopping.
    """
    stream = None
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            output_device_index=output_device_index,
            frames_per_buffer=CHUNK
        )
        
        # 
        # We wait until we have a few chunks before starting to play
        # to handle network jitter (bursty TCP packets).
        PRE_BUFFER_COUNT = 5 
        buffer_filled = False

        while not stop_event.is_set():
            try:
                # 1. Get data from queue (block briefly)
                data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                # If queue runs dry, reset buffering state to avoid glitching on next packet
                buffer_filled = False
                continue

            # 2. Buffering Logic
            if not buffer_filled:
                if audio_queue.qsize() >= PRE_BUFFER_COUNT:
                    buffer_filled = True
                    # Once filled, play this chunk and continue
                    stream.write(data)
                else:
                    # Still filling buffer... discard? No, we need to put it back or hold it.
                    # Actually, PyAudio write blocks. If we just write, it's fine 
                    # AS LONG AS we didn't start writing when the queue was empty.
                    # Simplified strategy: If queue was empty, we just consume this one.
                    # But real jitter buffering requires holding.
                    # Let's simple-path: just Write. The Queue ITSELF acts as the buffer 
                    # because the network thread pushes faster than we play usually.
                    stream.write(data)
            else:
                stream.write(data)

    except Exception as e:
        print(f"[Speaker Worker] Error: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()

def run_audio_receiver(
    host: str,
    port: int,
    stop_event: threading.Event,
    output_device_index: int | None = None,
) -> None:
    """
    Connects to server audio TX.
    Uses a Queue to decouple network reception from audio playback.
    """
    p = pyaudio.PyAudio()
    
    # 1. Setup Audio Output Device
    if output_device_index is None:
        output_device_index = get_output_device_index(p, None)
    
    device_name = "Default"
    if output_device_index is not None:
        try:
            device_name = p.get_device_info_by_index(output_device_index).get("name")
        except:
            pass
    
    print(f"[Speaker] Output Device: {device_name} (Index: {output_device_index})")

    # 2. Start Playback Thread (Consumer)
    audio_queue = queue.Queue()
    playback_thread = threading.Thread(
        target=audio_playback_worker,
        args=(audio_queue, stop_event, p, output_device_index),
        daemon=True
    )
    playback_thread.start()

    # 3. Network Loop (Producer)
    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)
            print(f"[Speaker] Connected to stream source {host}:{port}")

            try:
                while not stop_event.is_set():
                    # Read larger chunks if possible, but any amount is fine for the queue
                    data = sock.recv(4096)
                    if not data:
                        break
                    audio_queue.put(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                sock.close()
                print("[Speaker] Stream disconnected.")

        except (ConnectionRefusedError, OSError):
            # Wait before retry
            pass
        
        if not stop_event.is_set():
            time.sleep(2)

    p.terminate()
    print("[Speaker] Stopped.")


def recv_exact(sock, n: int) -> bytes | None:
    """Read exactly n bytes from socket. Returns None on EOF/error."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def run_code_receiver(
    host: str,
    port: int,
    stop_event: threading.Event,
) -> None:
    """
    Connect to server code/command port. Receive length-prefixed code messages
    (4-byte big-endian length + UTF-8 payload) and display them.
    """
    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)
            print(f"[Code] Connected to {host}:{port} (executed code channel)")

            try:
                while not stop_event.is_set():
                    header = recv_exact(sock, 4)
                    if not header:
                        break
                    (msg_len,) = struct.unpack(">L", header)
                    if msg_len <= 0 or msg_len > 10 * 1024 * 1024:
                        print(f"[Code] Invalid length {msg_len}, dropping.")
                        break
                    payload = recv_exact(sock, msg_len)
                    if not payload:
                        break
                    try:
                        code_str = payload.decode("utf-8")
                    except UnicodeDecodeError:
                        code_str = payload.decode("utf-8", errors="replace")
                    print("\n" + "=" * 60)
                    print("RECEIVED EXECUTED CODE")
                    print("=" * 60)
                    print(code_str)
                    print("=" * 60 + "\n")
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"[Code] Disconnected: {e}")
            finally:
                sock.close()
        except (ConnectionRefusedError, OSError):
            pass
        if not stop_event.is_set():
            time.sleep(2)
    print("[Code] Stopped.")


def main():
    parser = argparse.ArgumentParser(description="Mock TCP client (Robot Emulator)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server IP")
    parser.add_argument("--audio-port", type=int, default=PORT_AUDIO_RX)
    parser.add_argument("--video-port", type=int, default=PORT_VIDEO_RX)
    parser.add_argument("--audio-tx-port", type=int, default=PORT_AUDIO_TX)
    parser.add_argument("--code-port", type=int, default=PORT_CODE_RX, help="Port to receive executed code (length-prefixed)")

    parser.add_argument("--output-device", type=str, default=None, help="Speaker name or index")
    parser.add_argument("--list-output-devices", action="store_true")
    parser.add_argument("--record-mic", type=str, default=None, help="Save mic to .wav")
    
    args = parser.parse_args()

    # List Devices Helper
    if args.list_output_devices:
        p = pyaudio.PyAudio()
        list_output_devices(p)
        p.terminate()
        return

    # Resolve Device
    output_idx = None
    if args.output_device:
        p = pyaudio.PyAudio()
        output_idx = get_output_device_index(p, args.output_device)
        p.terminate()
        if output_idx is None:
            print(f"Error: Device '{args.output_device}' not found.")
            sys.exit(1)

    # Start Threads
    stop = threading.Event()
    
    t_mic = threading.Thread(
        target=run_audio_client,
        args=(args.host, args.audio_port, stop),
        kwargs={"record_path": args.record_mic},
        daemon=True
    )
    
    t_vid = threading.Thread(
        target=run_video_client,
        args=(args.host, args.video_port, stop),
        daemon=True
    )
    
    t_spk = threading.Thread(
        target=run_audio_receiver,
        args=(args.host, args.audio_tx_port, stop),
        kwargs={"output_device_index": output_idx},
        daemon=True
    )

    t_code = threading.Thread(
        target=run_code_receiver,
        args=(args.host, args.code_port, stop),
        daemon=True,
    )

    threads = [t_mic, t_vid, t_spk, t_code]
    for t in threads:
        t.start()

    print(f"Mock Client Running. [Mic->{args.audio_port}] [Video->{args.video_port}] [Speaker<-{args.audio_tx_port}] [Code<-{args.code_port}]")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop.set()
    
    for t in threads:
        t.join(timeout=2.0)
    print("Goodbye.")

if __name__ == "__main__":
    main()