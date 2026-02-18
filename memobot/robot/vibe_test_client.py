#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import socket
import struct
import threading
import subprocess
import argparse
import time
import os
import signal
import numpy as np
import cv2

# Naoqi modules (Standard on Nao Robot)
from naoqi import ALProxy, ALBroker

# --- CONFIGURATION ---
CHUNK_SIZE = 4096
TEMP_FILENAME = "/tmp/nao_exec_script.py"

# --- GLOBAL STATE ---
CURRENT_PROCESS = None
PROCESS_LOCK = threading.Lock()

# --- HELPER: SEND FEEDBACK ---
def send_feedback(server_ip, port, message):
    """Sends logs back to Gemini Server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((server_ip, port))
        payload = str(message)
        header = struct.pack(">L", len(payload))
        s.sendall(header + payload)
        s.close()
    except Exception as e:
        print "[Feedback] Failed to send logs: %s" % e

# --- HELPER: PROCESS MANAGEMENT ---
def kill_current_process():
    """Kills the running generated code subprocess safely."""
    global CURRENT_PROCESS
    with PROCESS_LOCK:
        if CURRENT_PROCESS and CURRENT_PROCESS.poll() is None:
            print "\n[System] 🛑 Killing current robot action..."
            try:
                # Try graceful termination first
                os.kill(CURRENT_PROCESS.pid, signal.SIGTERM)
                time.sleep(0.5)
                # Force kill if still running
                if CURRENT_PROCESS.poll() is None:
                    os.kill(CURRENT_PROCESS.pid, signal.SIGKILL)
            except Exception as e:
                print "[System] Error killing process: %s" % e
            print "[System] Action stopped."
        else:
            print "[System] No action is currently running."

# --- THREAD 1: AUDIO STREAMER ---
def thread_stream_audio(server_ip, port, stop_event):
    print "[Audio] Connecting to %s:%d..." % (server_ip, port)
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, port))
            # Capture raw audio: 16kHz, Mono, S16_LE
            cmd = ['arecord', '-f', 'S16_LE', '-r', '16000', '-c', '1', '-t', 'raw']
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            while not stop_event.is_set():
                data = process.stdout.read(CHUNK_SIZE)
                if not data: break
                s.sendall(data)
            
            process.terminate()
            s.close()
        except Exception:
            time.sleep(2) # Silent retry

# --- THREAD 2: VIDEO STREAMER ---
def thread_stream_video(robot_ip, server_ip, port, stop_event):
    print "[Video] Connecting to %s:%d..." % (server_ip, port)
    video_proxy = None
    name_id = None
    
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, port))
            
            if not video_proxy:
                video_proxy = ALProxy("ALVideoDevice", robot_ip, 9559)
                # QVGA=1, RGB=11, FPS=10 (Lower FPS to save CPU for execution)
                name_id = video_proxy.subscribe("gemini_eye", 1, 11, 10)
            
            while not stop_event.is_set():
                al_img = video_proxy.getImageRemote(name_id)
                if al_img:
                    width, height = al_img[0], al_img[1]
                    array = al_img[6]
                    img_header = np.frombuffer(array, dtype=np.uint8)
                    img = img_header.reshape((height, width, 3))
                    
                    # RGB -> BGR -> JPEG
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    
                    if result:
                        data = encimg.tostring()
                        s.sendall(struct.pack(">L", len(data)) + data)
                
                time.sleep(0.1) 
                
        except Exception:
            if video_proxy and name_id:
                try: video_proxy.unsubscribe(name_id)
                except: pass
                video_proxy = None
            time.sleep(2)

# --- THREAD 3: COMMAND EXECUTOR (SUBPROCESS MODE) ---
def thread_receive_commands(server_ip, cmd_port, feedback_port, stop_event):
    global CURRENT_PROCESS
    print "[Command] Connecting to %s:%d..." % (server_ip, cmd_port)
    
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, cmd_port))
            
            while not stop_event.is_set():
                # 1. Receive Code
                raw_len = s.recv(4)
                if not raw_len: break
                msg_len = struct.unpack(">L", raw_len)[0]
                
                code_data = b''
                while len(code_data) < msg_len:
                    packet = s.recv(msg_len - len(code_data))
                    if not packet: break
                    code_data += packet
                
                if not code_data: break

                # 2. Kill previous process if it's still running
                kill_current_process()

                print "\n" + "="*40
                print "EXECUTING REMOTE CODE (SUBPROCESS):"
                print "-"*40
                print code_data
                print "-"*40

                # 3. Write code to temp file
                # We inject a header to ensure encoding and basics are right
                header = "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\nimport time\nfrom naoqi import ALProxy\n\n"
                with open(TEMP_FILENAME, "w") as f:
                    f.write(header + code_data)

                # 4. Spawn Subprocess
                try:
                    with PROCESS_LOCK:
                        CURRENT_PROCESS = subprocess.Popen(
                            ['python', TEMP_FILENAME],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    
                    # Wait for completion (blocks THIS thread, not Main or Audio)
                    stdout, stderr = CURRENT_PROCESS.communicate()
                    
                    log_output = ""
                    if stdout: log_output += "[STDOUT]\n" + stdout
                    if stderr: log_output += "[STDERR]\n" + stderr
                    
                    print log_output

                    # 5. Send Feedback
                    send_feedback(server_ip, feedback_port, log_output)

                except Exception as e:
                    print "[Error] Execution failed: %s" % e
                    send_feedback(server_ip, feedback_port, "Execution Error: %s" % e)

        except Exception as e:
            print "[Command] Connection lost (%s). Retrying..." % e
            time.sleep(2)

# --- GLOBAL SIGNAL STATE ---
LAST_SIGINT_TIME = 0

def signal_handler(sig, frame):
    """
    Handles Ctrl+C.
    - First press: Kills the robot's current action (subprocess).
    - Second press (fast): Exits the client.
    """
    global LAST_SIGINT_TIME, stop_event
    current_time = time.time()
    
    # Check if this is a "Double Ctrl+C" (Force Quit)
    if (current_time - LAST_SIGINT_TIME) < 2.0:
        print "\n[System] 🛑 Force Quitting Client..."
        kill_current_process()
        stop_event.set()
        sys.exit(0)
    
    LAST_SIGINT_TIME = current_time
    
    # Standard logic: Stop the action, keep the client running
    if CURRENT_PROCESS and CURRENT_PROCESS.poll() is None:
        print "\n[Control] ⚠️ INTERRUPT RECEIVED: Stopping robot action..."
        kill_current_process()
        print "[Control] Client is still running. Waiting for next audio command..."
        print "[Control] (Press Ctrl+C again quickly to quit the client)"
    else:
        print "\n[Control] Client is idle. Press Ctrl+C again quickly to quit."

# --- MAIN ---
def main():
    global stop_event
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--server-ip", required=True)
    parser.add_argument("--port-audio-rx", type=int, default=50005)
    parser.add_argument("--port-video-rx", type=int, default=50006)
    parser.add_argument("--port-cmd-tx", type=int, default=50007)
    parser.add_argument("--port-feedback-rx", type=int, default=50008)
    args = parser.parse_args()

    stop_event = threading.Event()

    # Create Broker (keeps Naoqi happy)
    try:
        myBroker = ALBroker("myBroker", "0.0.0.0", 0, args.robot_ip, 9559)
    except: pass

    # Override the default Ctrl+C behavior
    signal.signal(signal.SIGINT, signal_handler)

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

    print "\n" + "*"*50
    print ">>> ROBOT CLIENT RUNNING <<<"
    print "Controls:"
    print "  [Ctrl+C] (Once)   -> KILL current executing robot action"
    print "  [Ctrl+C] (Twice)  -> QUIT this client"
    print "*"*50 + "\n"

    # Main loop just sleeps, waiting for signals
    # We use a loop with short sleep to allow signals to be caught immediately
    while not stop_event.is_set():
        time.sleep(0.5)

if __name__ == "__main__":
    main()