# Save as mac_master.py
import socket
import threading
import pyaudio
import cv2
import numpy as np
import struct
import sys
import time

# --- CONFIGURATION ---
HOST = '0.0.0.0'       # Listen on all interfaces
PORT_AUDIO_RX = 50005  # Receive Audio FROM Robot
PORT_VIDEO_RX = 50006  # Receive Video FROM Robot
PORT_AUDIO_TX = 50007  # Send Audio TO Robot

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def thread_receive_audio_from_robot():
    """Listens on 50005, receives audio from NAO, plays on Mac."""
    print(f"[Audio RX] Listening on {HOST}:{PORT_AUDIO_RX}...")
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    
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
            stream.write(data)
    except Exception as e:
        print(f"[Audio RX] Error: {e}")
    finally:
        if conn: conn.close()
        server_socket.close()
        stream.stop_stream()
        stream.close()
        p.terminate()

def thread_send_audio_to_robot():
    """Listens on 50007, reads Mac Mic, sends to NAO."""
    print(f"[Audio TX] Listening on {HOST}:{PORT_AUDIO_TX}...")
    
    p = pyaudio.PyAudio()
    # input=True means capture from Mic
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT_AUDIO_TX))
    server_socket.listen(1)
    
    conn = None
    try:
        conn, addr = server_socket.accept()
        print(f"[Audio TX] Robot Connected: {addr}")
        print("[Audio TX] Speaking into Mac Mic will now broadcast to Robot.")
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                conn.sendall(data)
            except IOError:
                pass # Ignore overflow errors
    except Exception as e:
        print(f"[Audio TX] Error: {e}")
    finally:
        if conn: conn.close()
        server_socket.close()
        stream.stop_stream()
        stream.close()
        p.terminate()

def recv_exact(sock, n):
    """Helper to receive exactly n bytes."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data += packet
    return data

def main_video_loop():
    """Listens on 50006, receives Video from NAO, displays it."""
    print(f"[Video RX] Listening on {HOST}:{PORT_VIDEO_RX}...")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT_VIDEO_RX))
    server_socket.listen(1)
    
    conn, addr = server_socket.accept()
    print(f"[Video RX] Robot Connected: {addr}")
    
    payload_size = struct.calcsize(">L")
    
    try:
        while True:
            # 1. Get Image Size
            packed_msg_size = recv_exact(conn, payload_size)
            if not packed_msg_size: break
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            
            # 2. Get Image Data
            frame_data = recv_exact(conn, msg_size)
            if not frame_data: break
            
            # 3. Decode & Display
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("NAO Robot Stream", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"[Video RX] Error: {e}")
    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start Audio threads
    t1 = threading.Thread(target=thread_receive_audio_from_robot)
    t2 = threading.Thread(target=thread_send_audio_to_robot)
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    
    # Run Video in main thread
    main_video_loop()
