robot


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Save as robot_av_client.py

import sys
import time
import socket
import struct
import argparse
import numpy as np
import cv2
from naoqi import ALProxy, ALBroker, ALModule
import vision_definitions

# --- GLOBALS ---
AudioModule = None

class AudioStreamerModule(ALModule):
    """
    NAOqi Module to capture audio and send it via TCP.
    """
    def __init__(self, name, client_socket):
        ALModule.__init__(self, name)
        self.client_socket = client_socket

    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, timestamp, buffer):
        """
        Callback from ALAudioDevice. 
        'buffer' is a string of 16-bit PCM data.
        """
        try:
            self.client_socket.sendall(buffer)
        except Exception as e:
            # If socket fails, we print but don't crash, 
            # though usually we should stop streaming.
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="127.0.0.1", help="Robot IP")
    parser.add_argument("--mac-ip", required=True, help="Mac Server IP")
    parser.add_argument("--audio-port", type=int, default=50005)
    parser.add_argument("--video-port", type=int, default=50006)
    parser.add_argument("--camera-id", type=int, default=0, help="0=Top, 1=Bottom")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. SETUP SOCKETS (One for Audio, One for Video)
    # ---------------------------------------------------------
    print "Connecting to Mac Audio Server %s:%d..." % (args.mac_ip, args.audio_port)
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_audio.connect((args.mac_ip, args.audio_port))

    print "Connecting to Mac Video Server %s:%d..." % (args.mac_ip, args.video_port)
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_video.connect((args.mac_ip, args.video_port))

    # ---------------------------------------------------------
    # 2. SETUP AUDIO (Background Callback)
    # ---------------------------------------------------------
    # We need a broker because we are creating an ALModule
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, args.robot_ip, 9559)

    global AudioModule
    AudioModule = AudioStreamerModule("AudioModule", sock_audio)

    audio_proxy = ALProxy("ALAudioDevice", args.robot_ip, 9559)
    # 16000Hz, Channel 3 (Front), 0 (Deinterleaved/Raw)
    audio_proxy.setClientPreferences(AudioModule.getName(), 16000, 3, 0)
    audio_proxy.subscribe(AudioModule.getName())
    print "Audio streaming started (Background)."

    # ---------------------------------------------------------
    # 3. SETUP VIDEO (Main Loop)
    # ---------------------------------------------------------
    video_proxy = ALProxy("ALVideoDevice", args.robot_ip, 9559)
    resolution = vision_definitions.kQVGA
    color_space = vision_definitions.kBGRColorSpace
    fps = 15
    
    name_id = video_proxy.subscribe("python_av_stream", resolution, color_space, fps)
    video_proxy.setParam(vision_definitions.kCameraSelectID, args.camera_id)
    print "Video streaming started (Main Loop)."
    print "Press Ctrl+C to stop both streams."

    try:
        while True:
            # --- Video Capture & Send ---
            al_img = video_proxy.getImageRemote(name_id)
            
            if al_img and al_img[6]:
                width = al_img[0]
                height = al_img[1]
                raw_data = al_img[6]
                
                # Convert to numpy (Use fromstring for Python 2.7/Old Numpy)
                frame = np.fromstring(raw_data, dtype=np.uint8).reshape((height, width, 3))

                # Compress to JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                result, encimg = cv2.imencode('.jpg', frame, encode_param)
                
                if result:
                    # Use tostring() for Python 2.7 compat
                    data = encimg.tostring()
                    size = len(data)
                    
                    # Send: [Size 4 bytes] + [JPEG Data]
                    sock_video.sendall(struct.pack(">L", size) + data)

            # Control framerate slightly
            time.sleep(0.01)

    except KeyboardInterrupt:
        print "\nStopping..."
    except Exception as e:
        print "Error: %s" % e
    finally:
        # Cleanup Audio
        try:
            audio_proxy.unsubscribe(AudioModule.getName())
        except: 
            pass
        
        # Cleanup Video
        try:
            video_proxy.unsubscribe(name_id)
        except:
            pass

        # Cleanup Sockets
        sock_audio.close()
        sock_video.close()
        myBroker.shutdown()
        print "Shutdown complete."

if __name__ == "__main__":
    main()



—-----------------------------------------------------------------------------------------------------------

Server 


# Save as mac_av_server.py
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
