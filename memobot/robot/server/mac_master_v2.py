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

# --- ECHO CANCELLATION CONFIG ---
# Threshold: If Mac Mic RMS is > this, we assume you are talking.
# Adjust this if it cuts out too much (increase) or allows echo (decrease).
VOICE_THRESHOLD = 800 

# Latency Buffer: How long to keep muting AFTER you stop talking.
# This covers the time it takes for your voice to travel to the robot and back.
ECHO_LATENCY_SECONDS = 0.9 

# Shared State
state = {
    "mute_until": 0.0
}

def calculate_rms(data):
    """Calculates the root mean square (volume) of the audio chunk."""
    try:
        # Convert raw bytes to numpy array of 16-bit integers
        shorts = np.frombuffer(data, dtype=np.int16)
        # Avoid math domain error for empty buffer
        if len(shorts) == 0:
            return 0
        # RMS calculation
        return np.sqrt(np.mean(shorts.astype(np.float64)**2))
    except Exception:
        return 0

def thread_receive_audio_from_robot():
    """Listens on 50005, receives audio from NAO, plays on Mac (with Echo Suppression)."""
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
        
        # Create a block of silence to play when muting
        silence = b'\x00' * CHUNK * 2

        while True:
            data = conn.recv(CHUNK)
            if not data: break
            
            # --- ECHO SUPPRESSION LOGIC ---
            # If we are currently talking (or recently stopped), mute the incoming audio
            if time.time() < state["mute_until"]:
                # Play silence or reduce volume significantly
                stream.write(silence[0:len(data)])
            else:
                # Play actual robot audio
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
    """Listens on 50007, reads Mac Mic, sends to NAO, and detects Voice Activity."""
    print(f"[Audio TX] Listening on {HOST}:{PORT_AUDIO_TX}...")
    
    p = pyaudio.PyAudio()
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
                # Read Mic Data
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # --- VOICE ACTIVITY DETECTION ---
                rms = calculate_rms(data)
                
                if rms > VOICE_THRESHOLD:
                    # You are talking. Update the mute timer.
                    # We add ECHO_LATENCY_SECONDS to the current time.
                    state["mute_until"] = time.time() + ECHO_LATENCY_SECONDS
                
                # Send data to robot
                conn.sendall(data)
                
            except IOError:
                pass 
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
            packed_msg_size = recv_exact(conn, payload_size)
            if not packed_msg_size: break
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            
            frame_data = recv_exact(conn, msg_size)
            if not frame_data: break
            
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                # Visual Indicator of Mute State
                if time.time() < state["mute_until"]:
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) # Red Dot = Echo Suppression Active
                
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
    t1 = threading.Thread(target=thread_receive_audio_from_robot)
    t2 = threading.Thread(target=thread_send_audio_to_robot)
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    
    main_video_loop()
