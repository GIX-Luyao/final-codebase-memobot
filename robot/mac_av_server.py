import socket
import threading
import pyaudio
import cv2
import numpy as np
import struct
import sys

# --- CONFIGURATION ---
HOST = '0.0.0.0'       # Listen on all interfaces
AUDIO_PORT = 50005     # Port for Audio Stream
VIDEO_PORT = 50006     # Port for Video Stream

# Audio Config (Must match NAO)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096

def audio_server_thread():
    """
    Handles incoming audio stream from NAO and plays it.
    """
    print(f"[Audio] Listening on {HOST}:{AUDIO_PORT}...")
    
    # Setup PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, AUDIO_PORT))
        server_socket.listen(1)
        conn, addr = server_socket.accept()
        print(f"[Audio] NAO Connected from {addr}")

        while True:
            data = conn.recv(CHUNK)
            if not data:
                break
            stream.write(data)
            
    except Exception as e:
        print(f"[Audio] Error: {e}")
    finally:
        print("[Audio] Closing stream...")
        if 'conn' in locals(): conn.close()
        server_socket.close()
        stream.stop_stream()
        stream.close()
        p.terminate()

def recv_exact(sock, n):
    """Helper to receive exactly n bytes for video framing."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def video_server_loop():
    """
    Handles incoming video stream and displays it (Main Thread).
    """
    print(f"[Video] Listening on {HOST}:{VIDEO_PORT}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, VIDEO_PORT))
        server_socket.listen(1)
        conn, addr = server_socket.accept()
        print(f"[Video] NAO Connected from {addr}")
        
        payload_size = struct.calcsize(">L")

        while True:
            # 1. Read Image Size
            packed_msg_size = recv_exact(conn, payload_size)
            if not packed_msg_size:
                break
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            
            # 2. Read Image Data
            frame_data = recv_exact(conn, msg_size)
            if not frame_data:
                break

            # 3. Decode & Show
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("NAO Live Stream (Video + Audio)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"[Video] Error: {e}")
    finally:
        print("[Video] Closing...")
        if 'conn' in locals(): conn.close()
        server_socket.close()
        cv2.destroyAllWindows()

def main():
    # Start Audio in a separate thread (so it doesn't block video)
    t_audio = threading.Thread(target=audio_server_thread)
    t_audio.daemon = True # Kills thread if main program quits
    t_audio.start()

    # Run Video in main thread (OpenCV requires main thread for GUI)
    video_server_loop()

if __name__ == "__main__":
    main()
