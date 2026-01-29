# mac_master.py
# Runs main flow: capture one frame from NAO → recognize user → run Realtime API.
# Audio I/O is bridged: NAO mic → Realtime API input, Realtime API output → NAO speaker.
# NAO connects as TCP client to this server.

import asyncio
import queue
import socket
import struct
import sys
import threading
from pathlib import Path

import numpy as np

# Project root and query_pipeline for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT_AUDIO_RX = 50005  # Receive audio FROM NAO (robot mic → API input)
PORT_VIDEO_RX = 50006  # Receive video FROM NAO (one frame used for face recognition)
PORT_AUDIO_TX = 50007  # Send audio TO NAO (API output → robot speaker)

# NAO sends 16 kHz; Realtime API uses 24 kHz
NAO_RATE = 16000
API_RATE = 24000
CHUNK_NAO = 1024
BYTES_PER_SAMPLE = 2


def resample_16k_to_24k(data: bytes) -> bytes:
    """Resample 16 kHz mono 16-bit PCM to 24 kHz (for API input)."""
    if not data:
        return data
    n = len(data) // BYTES_PER_SAMPLE
    arr = np.frombuffer(data, dtype=np.int16)
    out_len = int(n * API_RATE / NAO_RATE)
    x_old = np.arange(n)
    x_new = np.linspace(0, n - 1, out_len)
    out = np.interp(x_new, x_old, arr.astype(np.float64)).astype(np.int16)
    return out.tobytes()


def resample_24k_to_16k(data: bytes) -> bytes:
    """Resample 24 kHz mono 16-bit PCM to 16 kHz (for NAO output)."""
    if not data:
        return data
    n = len(data) // BYTES_PER_SAMPLE
    arr = np.frombuffer(data, dtype=np.int16)
    out_len = int(n * NAO_RATE / API_RATE)
    x_old = np.arange(n)
    x_new = np.linspace(0, n - 1, out_len)
    out = np.interp(x_new, x_old, arr.astype(np.float64)).astype(np.int16)
    return out.tobytes()


def recv_exact_sync(sock, n):
    """Receive exactly n bytes from a socket (sync)."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def capture_one_frame_sync(video_sock) -> bytes:
    """Read one length-prefixed image from video socket. Returns raw image bytes or None."""
    payload_size = struct.calcsize(">L")
    packed = recv_exact_sync(video_sock, payload_size)
    if not packed:
        return None
    msg_size = struct.unpack(">L", packed)[0]
    frame_data = recv_exact_sync(video_sock, msg_size)
    return frame_data


async def main():
    # Load env (robo_client loads .env; ensure we have it for OPENAI_API_KEY)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    api_key = __import__("os").environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        print("ERROR: OPENAI_API_KEY is not set or invalid. Set it in .env")
        sys.exit(1)

    print("[mac_master] Starting TCP servers for NAO (client)...")
    print(f"  Audio RX (robot mic):  {HOST}:{PORT_AUDIO_RX}")
    print(f"  Video RX (one frame):  {HOST}:{PORT_VIDEO_RX}")
    print(f"  Audio TX (robot spk):  {HOST}:{PORT_AUDIO_TX}")
    print("  Waiting for NAO to connect to all three...")

    # Use blocking TCP servers so we can share sockets with sync frame read
    server_audio_rx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_audio_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_audio_rx.bind((HOST, PORT_AUDIO_RX))
    server_audio_rx.listen(1)

    server_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_video.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_video.bind((HOST, PORT_VIDEO_RX))
    server_video.listen(1)

    server_audio_tx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_audio_tx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_audio_tx.bind((HOST, PORT_AUDIO_TX))
    server_audio_tx.listen(1)

    # Accept NAO connections (run in executor to not block event loop)
    loop = asyncio.get_event_loop()

    def accept_connections():
        conn_audio_rx, _ = server_audio_rx.accept()
        conn_video, _ = server_video.accept()
        conn_audio_tx, _ = server_audio_tx.accept()
        return conn_audio_rx, conn_video, conn_audio_tx

    conn_audio_rx, conn_video, conn_audio_tx = await loop.run_in_executor(None, accept_connections)
    print("[mac_master] NAO connected to all ports.")

    # Capture one frame from NAO video and save for recognition
    image_path = ROOT / "query_pipeline" / "image.png"
    frame_data = await loop.run_in_executor(None, capture_one_frame_sync, conn_video)
    if not frame_data:
        print("[mac_master] No video frame received; using existing image.png if present.")
    else:
        import cv2
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imwrite(str(image_path), frame)
            print(f"[mac_master] Saved one frame to {image_path}")
        # Close video connection; we only needed one frame
        try:
            conn_video.close()
        except Exception:
            pass

    # Recognize user (sync, same as main.py)
    from query_pipeline.recognize_user import recognize_user
    print(f"[mac_master] Recognizing user from {image_path}...")
    try:
        person = recognize_user(image_path=str(image_path))
        user_name = person["name"]
        print(f"[main] Recognized: {user_name}. Starting Realtime API with robot audio I/O.\n")
    except SystemExit:
        user_name = None
        print("[main] Recognition failed or no image; continuing without user name.\n")
    except Exception as e:
        print(f"[main] Recognition error: {e}; continuing without user name.\n")
        user_name = None

    # Queue-based bridge: NAO 16 kHz ↔ Realtime API 24 kHz
    # RX: thread reads from NAO → asyncio.Queue → async generator yields resampled 24k to API
    # TX: async callback puts resampled 16k → queue.Queue → thread sends to NAO
    audio_rx_queue = asyncio.Queue()
    audio_tx_queue = queue.Queue()

    def read_audio_rx_loop():
        try:
            while True:
                data = conn_audio_rx.recv(4096)
                if not data:
                    break
                audio_rx_queue.put_nowait(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                audio_rx_queue.put_nowait(None)
            except Exception:
                pass

    def write_audio_tx_loop():
        try:
            while True:
                data = audio_tx_queue.get()
                if data is None:
                    break
                conn_audio_tx.sendall(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    rx_thread = threading.Thread(target=read_audio_rx_loop)
    rx_thread.daemon = True
    rx_thread.start()

    tx_thread = threading.Thread(target=write_audio_tx_loop)
    tx_thread.daemon = True
    tx_thread.start()

    async def robot_audio_reader():
        """Async generator: read 16 kHz PCM from NAO, resample to 24 kHz, yield for API."""
        while True:
            chunk = await audio_rx_queue.get()
            if chunk is None:
                break
            resampled = resample_16k_to_24k(chunk)
            if resampled:
                yield resampled

    def output_audio_callback(audio_24k: bytes):
        out = resample_24k_to_16k(audio_24k)
        if out:
            audio_tx_queue.put(out)

    # Run Realtime API with robot bridge (same event loop)
    from query_pipeline.robo_client import run_realtime_with_robot_async
    print("\n🎙️  Memobot Realtime (audio via NAO)\n")
    try:
        await run_realtime_with_robot_async(
            user_name=user_name,
            robot_audio_reader=robot_audio_reader(),
            output_audio_callback=output_audio_callback,
        )
    except KeyboardInterrupt:
        print("\n[mac_master] Interrupted.")
    finally:
        try:
            audio_tx_queue.put(None)
        except Exception:
            pass
        try:
            conn_audio_rx.close()
            conn_audio_tx.close()
        except Exception:
            pass
        server_audio_rx.close()
        server_video.close()
        server_audio_tx.close()


if __name__ == "__main__":
    asyncio.run(main())
