#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import socket
import struct
import threading
import subprocess
import argparse
import time
<<<<<<< HEAD
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
=======
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
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
        payload = str(message)
        header = struct.pack(">L", len(payload))
        s.sendall(header + payload)
        s.close()
    except Exception as e:
<<<<<<< HEAD
        print "[Feedback] Failed to send logs to server: %s" % e

# --- THREAD 1: AUDIO STREAMER (ROBOT MIC -> SERVER) ---
=======
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
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
def thread_stream_audio(server_ip, port, stop_event):
    print "[Audio] Connecting to %s:%d..." % (server_ip, port)
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, port))
<<<<<<< HEAD
            
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
=======
            # Capture raw audio: 16kHz, Mono, S16_LE
            cmd = ['arecord', '-f', 'S16_LE', '-r', '16000', '-c', '1', '-t', 'raw']
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            while not stop_event.is_set():
                data = process.stdout.read(CHUNK_SIZE)
                if not data: break
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
                s.sendall(data)
            
            process.terminate()
            s.close()
<<<<<<< HEAD
        except Exception as e:
            print "[Audio] Connection lost (%s). Retrying in 2s..." % e
            time.sleep(2)

# --- THREAD 2: VIDEO STREAMER (ROBOT CAM -> SERVER) ---
=======
        except Exception:
            time.sleep(2) # Silent retry

# --- THREAD 2: VIDEO STREAMER ---
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
def thread_stream_video(robot_ip, server_ip, port, stop_event):
    print "[Video] Connecting to %s:%d..." % (server_ip, port)
    video_proxy = None
    name_id = None
    
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, port))
            
<<<<<<< HEAD
            # Setup Naoqi Video
            if not video_proxy:
                video_proxy = ALProxy("ALVideoDevice", robot_ip, 9559)
                # Subscribe: Name, Resolution (1=QVGA), ColorSpace (11=RGB), FPS
                name_id = video_proxy.subscribe("python_streamer", 1, 11, 15)
            
            print "[Video] Streaming started."
=======
            if not video_proxy:
                video_proxy = ALProxy("ALVideoDevice", robot_ip, 9559)
                # QVGA=1, RGB=11, FPS=10 (Lower FPS to save CPU for execution)
                name_id = video_proxy.subscribe("gemini_eye", 1, 11, 10)
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
            
            while not stop_event.is_set():
                al_img = video_proxy.getImageRemote(name_id)
                if al_img:
<<<<<<< HEAD
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
=======
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
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
            if video_proxy and name_id:
                try: video_proxy.unsubscribe(name_id)
                except: pass
                video_proxy = None
            time.sleep(2)

<<<<<<< HEAD
# --- THREAD 3: COMMAND EXECUTOR (SERVER -> ROBOT) ---
def thread_receive_commands(server_ip, cmd_port, feedback_port, stop_event):
    print "[Command] Connecting to %s:%d..." % (server_ip, cmd_port)
    
    # Pre-load imports into the execution globals
    global EXEC_GLOBALS
    EXEC_GLOBALS['ALProxy'] = ALProxy
    EXEC_GLOBALS['ALBroker'] = ALBroker
    EXEC_GLOBALS['sys'] = sys
    EXEC_GLOBALS['time'] = time
    
=======
# --- THREAD 3: COMMAND EXECUTOR (SUBPROCESS MODE) ---
def thread_receive_commands(server_ip, cmd_port, feedback_port, stop_event):
    global CURRENT_PROCESS
    print "[Command] Connecting to %s:%d..." % (server_ip, cmd_port)
    
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
    while not stop_event.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, cmd_port))
            
            while not stop_event.is_set():
<<<<<<< HEAD
                # 1. Read Header (4 bytes length)
=======
                # 1. Receive Code
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
                raw_len = s.recv(4)
                if not raw_len: break
                msg_len = struct.unpack(">L", raw_len)[0]
                
<<<<<<< HEAD
                # 2. Read Payload (Code)
=======
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
                code_data = b''
                while len(code_data) < msg_len:
                    packet = s.recv(msg_len - len(code_data))
                    if not packet: break
                    code_data += packet
                
                if not code_data: break

<<<<<<< HEAD
                print "\n" + "="*40
                print "EXECUTING REMOTE CODE:"
=======
                # 2. Kill previous process if it's still running
                kill_current_process()

                print "\n" + "="*40
                print "EXECUTING REMOTE CODE (SUBPROCESS):"
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
                print "-"*40
                print code_data
                print "-"*40

<<<<<<< HEAD
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
=======
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
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464
    parser.add_argument("--port-audio-rx", type=int, default=50005)
    parser.add_argument("--port-video-rx", type=int, default=50006)
    parser.add_argument("--port-cmd-tx", type=int, default=50007)
    parser.add_argument("--port-feedback-rx", type=int, default=50008)
    args = parser.parse_args()

    stop_event = threading.Event()

<<<<<<< HEAD
    # Create Broker (Required for some ALProxy functionality)
    try:
        myBroker = ALBroker("myBroker", "0.0.0.0", 0, args.robot_ip, 9559)
    except Exception as e:
        print "Could not connect to Naoqi at %s:9559" % args.robot_ip
        print "Error: %s" % e
        sys.exit(1)
=======
    # Create Broker (keeps Naoqi happy)
    try:
        myBroker = ALBroker("myBroker", "0.0.0.0", 0, args.robot_ip, 9559)
    except: pass

    # Override the default Ctrl+C behavior
    signal.signal(signal.SIGINT, signal_handler)
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464

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

<<<<<<< HEAD
    print "\n>>> CLIENT RUNNING. Press Ctrl+C to stop. <<<\n"

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print "\nStopping client..."
        stop_event.set()
        myBroker.shutdown()
        sys.exit(0)
=======
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
>>>>>>> 4739e9ef542ac4ee6788f75eea5d462a5ef2b464

if __name__ == "__main__":
    main()