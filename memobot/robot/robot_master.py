#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Save as robot_master.py

import sys
import time
import socket
import struct
import argparse
import threading
import subprocess
import numpy as np
import cv2
from naoqi import ALProxy, ALBroker, ALModule
import vision_definitions

# --- GLOBALS ---
AudioCaptureModule = None

# --- CLASSES ---

class AudioStreamerModule(ALModule):
    """ Captures audio from NAO mic and sends to Mac (Port 50005) """
    def __init__(self, name, client_socket):
        ALModule.__init__(self, name)
        self.client_socket = client_socket

    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, timestamp, buffer):
        try:
            self.client_socket.sendall(buffer)
        except Exception:
            pass

# --- THREADED FUNCTIONS ---

def thread_receive_audio_from_mac(mac_ip, port):
    """ Connects to Mac (Port 50007), receives audio, plays via 'aplay' """
    print "Connecting to Mac Audio TX (Speaker) at %s:%d..." % (mac_ip, port)
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((mac_ip, port))
        print "Connected to Mac Audio Source."
    except Exception as e:
        print "Failed to connect to Mac Audio Source: %s" % e
        return

    # Using aplay to pipe raw audio to speakers
    cmd = ['aplay', '-t', 'raw', '-r', '16000', '-f', 'S16_LE', '-c', '1']
    audio_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        while True:
            data = s.recv(4096)
            if not data:
                break
            audio_process.stdin.write(data)
    except Exception as e:
        print "Error processing incoming audio: %s" % e
    finally:
        s.close()
        try:
            audio_process.stdin.close()
            audio_process.terminate()
        except:
            pass
        print "Incoming audio thread stopped."

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--mac-ip", required=True)
    parser.add_argument("--port-audio-rx", type=int, default=50005) # Robot Sends -> Mac
    parser.add_argument("--port-video-rx", type=int, default=50006) # Robot Sends -> Mac
    parser.add_argument("--port-audio-tx", type=int, default=50007) # Robot Recvs <- Mac
    parser.add_argument("--camera-id", type=int, default=0)
    args = parser.parse_args()

    # 1. Start Incoming Audio Thread (Mac Mic -> Robot Speaker)
    t_audio_in = threading.Thread(target=thread_receive_audio_from_mac, 
                                  args=(args.mac_ip, args.port_audio_tx))
    t_audio_in.daemon = True
    t_audio_in.start()

    # 2. Setup Outgoing Audio (Robot Mic -> Mac Speaker)
    print "Connecting to Mac Audio RX (Mic Stream) at %s:%d..." % (args.mac_ip, args.port_audio_rx)
    sock_audio_out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock_audio_out.connect((args.mac_ip, args.port_audio_rx))
    except Exception as e:
        print "Could not connect to Mac Audio RX: %s" % e
        return

    # Init NAOqi Audio
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, args.robot_ip, 9559)
    global AudioCaptureModule
    AudioCaptureModule = AudioStreamerModule("AudioCaptureModule", sock_audio_out)
    
    audio_proxy = ALProxy("ALAudioDevice", args.robot_ip, 9559)
    audio_proxy.setClientPreferences(AudioCaptureModule.getName(), 16000, 3, 0)
    audio_proxy.subscribe(AudioCaptureModule.getName())
    
    # 3. Setup Outgoing Video (Robot Cam -> Mac Screen)
    print "Connecting to Mac Video RX at %s:%d..." % (args.mac_ip, args.port_video_rx)
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock_video.connect((args.mac_ip, args.port_video_rx))
    except Exception as e:
        print "Could not connect to Mac Video RX: %s" % e
        return

    video_proxy = ALProxy("ALVideoDevice", args.robot_ip, 9559)
    resolution = vision_definitions.kQVGA
    color_space = vision_definitions.kBGRColorSpace
    fps = 15
    name_id = video_proxy.subscribe("python_av_full", resolution, color_space, fps)
    video_proxy.setParam(vision_definitions.kCameraSelectID, args.camera_id)

    print "All streams running. Press Ctrl+C to stop."

    try:
        while True:
            # Video Loop
            al_img = video_proxy.getImageRemote(name_id)
            if al_img and al_img[6]:
                w, h = al_img[0], al_img[1]
                raw_data = al_img[6]
                
                # Use fromstring (Older Numpy)
                frame = np.fromstring(raw_data, dtype=np.uint8).reshape((h, w, 3))
                
                # Compress
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                result, encimg = cv2.imencode('.jpg', frame, encode_param)
                
                if result:
                    # Use tostring (Older Numpy)
                    data = encimg.tostring()
                    size = len(data)
                    sock_video.sendall(struct.pack(">L", size) + data)
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print "\nStopping..."
    finally:
        # Clean Audio Out
        try: audio_proxy.unsubscribe(AudioCaptureModule.getName())
        except: pass
        
        # Clean Video Out
        try: video_proxy.unsubscribe(name_id)
        except: pass
        
        # Clean Sockets
        sock_audio_out.close()
        sock_video.close()
        myBroker.shutdown()
        print "Shutdown complete."

if __name__ == "__main__":
    main()
