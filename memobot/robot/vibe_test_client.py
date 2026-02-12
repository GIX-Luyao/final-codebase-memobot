#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import socket
import struct
import threading
import subprocess
import argparse
import time
import traceback
from StringIO import StringIO
import numpy as np
import cv2

# Naoqi modules
from naoqi import ALProxy, ALBroker, ALModule
import vision_definitions

# --- CONFIGURATION ---
CHUNK_SIZE = 4096
# Persistent globals dictionary to maintain state between voice commands
# (e.g. if one command defines 'tts', the next command can use 'tts')
EXEC_GLOBALS = {}

# --- HELPER: SEND FEEDBACK ---
def send_feedback(server_ip, port, message):
    """Sends logs, errors, or success messages back to the server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((server_ip, port))
        
        # Protocol: [Length (4 bytes)] [Payload]
        payload = str(message)
        header = struct.pack(">L", len(payload))
        s.sendall(header + payload)
        s.close()
    except Exception as e:
        print "[Feedback] Failed to send logs to server: %s" % e

# --- THREAD 1: AUDIO STREAMER (ROBOT MIC -> SERVER) ---
def thread_stream_audio(server_ip, port, stop_event):
    print "[Audio] Connecting to %s:%d..." % (server_ip, port)
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, port))
            
            # Use 'arecord' to capture raw audio from the robot's mic
            # -f S16_LE: Signed 16-bit Little Endian
            # -r 16000: 16kHz sample rate
            # -c 1: Mono
            # -t raw: Raw PCM
            cmd = ['arecord', '-f', 'S16_LE', '-r', '16000', '-c', '1', '-t', 'raw']
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print "[Audio] Streaming started."
            while not stop_event.is_set():
                data = process.stdout.read(CHUNK_SIZE)
                if not data:
                    break
                s.sendall(data)
            
            process.terminate()
            s.close()
        except Exception as e:
            print "[Audio] Connection lost (%s). Retrying in 2s..." % e
            time.sleep(2)

# --- THREAD 2: VIDEO STREAMER (ROBOT CAM -> SERVER) ---
def thread_stream_video(robot_ip, server_ip, port, stop_event):
    print "[Video] Connecting to %s:%d..." % (server_ip, port)
    video_proxy = None
    name_id = None
    
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, port))
            
            # Setup Naoqi Video
            if not video_proxy:
                video_proxy = ALProxy("ALVideoDevice", robot_ip, 9559)
                # Subscribe: Name, Resolution (1=QVGA), ColorSpace (11=RGB), FPS
                name_id = video_proxy.subscribe("python_streamer", 1, 11, 15)
            
            print "[Video] Streaming started."
            
            while not stop_event.is_set():
                al_img = video_proxy.getImageRemote(name_id)
                if al_img:
                    width = al_img[0]
                    height = al_img[1]
                    array = al_img[6]
                    
                    # Convert raw buffer to numpy
                    img_header = np.frombuffer(array, dtype=np.uint8)
                    img = img_header.reshape((height, width, 3))
                    
                    # Convert RGB (Nao) to BGR (OpenCV standard)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Compress to JPEG
                    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if result:
                        data = encimg.tostring()
                        # Protocol: [Length (4 bytes)] [Payload]
                        s.sendall(struct.pack(">L", len(data)) + data)
                
                time.sleep(0.05) # Cap FPS slightly
                
        except Exception as e:
            print "[Video] Connection lost (%s). Retrying in 2s..." % e
            if video_proxy and name_id:
                try: video_proxy.unsubscribe(name_id)
                except: pass
                video_proxy = None
            time.sleep(2)

# --- THREAD 3: COMMAND EXECUTOR (SERVER -> ROBOT) ---
def thread_receive_commands(server_ip, cmd_port, feedback_port, stop_event):
    print "[Command] Connecting to %s:%d..." % (server_ip, cmd_port)
    
    # Pre-load imports into the execution globals
    global EXEC_GLOBALS
    EXEC_GLOBALS['ALProxy'] = ALProxy
    EXEC_GLOBALS['ALBroker'] = ALBroker
    EXEC_GLOBALS['sys'] = sys
    EXEC_GLOBALS['time'] = time
    
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, cmd_port))
            
            while not stop_event.is_set():
                # 1. Read Header (4 bytes length)
                raw_len = s.recv(4)
                if not raw_len: break
                msg_len = struct.unpack(">L", raw_len)[0]
                
                # 2. Read Payload (Code)
                code_data = b''
                while len(code_data) < msg_len:
                    packet = s.recv(msg_len - len(code_data))
                    if not packet: break
                    code_data += packet
                
                if not code_data: break

                print "\n" + "="*40
                print "EXECUTING REMOTE CODE:"
                print "-"*40
                print code_data
                print "-"*40

                # 3. Capture Stdout/Stderr
                capture = StringIO()
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = capture
                sys.stderr = capture
                
                error_status = "SUCCESS"
                
                try:
                    # EXECUTE THE CODE
                    # We use EXEC_GLOBALS to allow variables (like 'tts') to persist 
                    # between different voice commands.
                    exec(code_data, EXEC_GLOBALS)
                    
                except KeyboardInterrupt:
                    print "\n[Stopped by User]"
                    error_status = "INTERRUPTED"
                except Exception:
                    traceback.print_exc()
                    error_status = "ERROR"
                finally:
                    # Restore Output
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    
                    logs = capture.getvalue()
                    print logs # Print to local terminal
                    
                    # 4. Send Feedback if there is output or error
                    if len(logs) > 0 or error_status != "SUCCESS":
                        full_report = "[STATUS: %s]\n%s" % (error_status, logs)
                        send_feedback(server_ip, feedback_port, full_report)
                        print "[Feedback] Sent report to server."

        except Exception as e:
            print "[Command] Connection lost (%s). Retrying in 2s..." % e
            time.sleep(2)

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="127.0.0.1", help="IP of the robot (localhost if running on robot)")
    parser.add_argument("--server-ip", required=True, help="IP of the Mac/Server running Gemini logic")
    parser.add_argument("--port-audio-rx", type=int, default=50005)
    parser.add_argument("--port-video-rx", type=int, default=50006)
    parser.add_argument("--port-cmd-tx", type=int, default=50007)
    parser.add_argument("--port-feedback-rx", type=int, default=50008)
    args = parser.parse_args()

    stop_event = threading.Event()

    # Create Broker (Required for some ALProxy functionality)
    try:
        myBroker = ALBroker("myBroker", "0.0.0.0", 0, args.robot_ip, 9559)
    except Exception as e:
        print "Could not connect to Naoqi at %s:9559" % args.robot_ip
        print "Error: %s" % e
        sys.exit(1)

    # Start Threads
    t_audio = threading.Thread(target=thread_stream_audio, args=(args.server_ip, args.port_audio_rx, stop_event))
    t_video = threading.Thread(target=thread_stream_video, args=(args.robot_ip, args.server_ip, args.port_video_rx, stop_event))
    t_cmd   = threading.Thread(target=thread_receive_commands, args=(args.server_ip, args.port_cmd_tx, args.port_feedback_rx, stop_event))

    t_audio.daemon = True
    t_video.daemon = True
    t_cmd.daemon = True

    t_audio.start()
    t_video.start()
    t_cmd.start()

    print "\n>>> CLIENT RUNNING. Press Ctrl+C to stop. <<<\n"

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print "\nStopping client..."
        stop_event.set()
        myBroker.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()