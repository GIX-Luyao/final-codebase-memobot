#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import socket
import struct
import argparse
import threading
import subprocess
import select
import numpy as np
import cv2
from naoqi import ALProxy, ALBroker, ALModule
import vision_definitions

# --- GLOBALS ---
AudioCaptureModule = None
KeywordModule = None
STATE_STREAMING = False
STATE_RUNNING = True
mutex = threading.Lock()

# Resources
resources = {
    "sock_audio_out": None, "sock_video": None, "thread_audio_in": None,
    "audio_proxy": None, "video_proxy": None, "video_name_id": None,
    "life_proxy": None,
    "stop_audio_thread_event": None
}

# --- CLASSES ---

class AudioStreamerModule(ALModule):
    def __init__(self, name, client_socket):
        ALModule.__init__(self, name)
        self.client_socket = client_socket

    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, timestamp, buffer):
        if self.client_socket:
            try:
                self.client_socket.sendall(buffer)
            except Exception:
                pass

class KeywordService(ALModule):
    def __init__(self, name):
        ALModule.__init__(self, name)
        self.memory = ALProxy("ALMemory")
        self.asr = ALProxy("ALSpeechRecognition")
        self.tts = ALProxy("ALTextToSpeech")
        
        # 1. Setup Vocabulary (Simplified to just "start" and "stop")
        self.asr.pause(True)
        try:
            self.asr.setLanguage("English")
            vocab = ["start", "stop"]
            self.asr.setVocabulary(vocab, True) 
            print "[ASR] Vocabulary set to: %s" % vocab
        except Exception as e:
            print "[ASR] Setup Error: %s" % e
        self.asr.pause(False)
        
        # 2. Subscribe
        self.memory.subscribeToEvent("WordRecognized", self.getName(), "onWordRecognized")
        self.asr.subscribe("KeywordService_ASR")
        
        # 3. Ready Confirmation
        self.tts.say("Ready")

    def onWordRecognized(self, key, value, message):
        global STATE_STREAMING
        
        try:
            # Expected structure: [ "word_string", confidence_float ]
            if len(value) >= 2 and isinstance(value[1], float):
                word = value[0]
                conf = value[1]
                
                # Debug only if confidence is decent (reduces log spam)
                if conf > 0.3:
                    print "[DEBUG] Heard: '%s' (Confidence: %.2f)" % (word, conf)

                # Threshold set to 0.4 based on your logs
                if conf > 0.4: 
                    with mutex:
                        # TRIGGER: START
                        if word == "start" and not STATE_STREAMING:
                            print "\n>>> COMMAND RECEIVED: START <<<"
                            self.tts.say("Starting stream")
                            STATE_STREAMING = True
                        
                        # TRIGGER: STOP
                        elif word == "stop" and STATE_STREAMING:
                            print "\n>>> COMMAND RECEIVED: STOP <<<"
                            self.tts.say("Stopping stream")
                            STATE_STREAMING = False
        except Exception as e:
            print "[ERROR] parsing word: %s" % e

    def shutdown(self):
        try:
            self.asr.unsubscribe("KeywordService_ASR")
            self.memory.unsubscribeToEvent("WordRecognized", self.getName())
        except:
            pass

# --- THREADED FUNCTIONS ---

def thread_receive_audio_from_mac(mac_ip, port, stop_event):
    print "Connecting to Mac Audio TX..."
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect((mac_ip, port))
    except Exception as e:
        print "Failed to connect to Mac Audio Source: %s" % e
        return

    cmd = ['aplay', '-t', 'raw', '-r', '16000', '-f', 'S16_LE', '-c', '1']
    audio_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        while not stop_event.is_set():
            try:
                data = s.recv(4096)
                if not data: break
                audio_process.stdin.write(data)
            except socket.timeout: continue
            except socket.error: break
    except Exception: pass
    finally:
        s.close()
        try: audio_process.terminate()
        except: pass

# --- HELPER FUNCTIONS ---

def start_services(args, robot_ip):
    global AudioCaptureModule, resources
    
    # 1. Start Incoming Audio
    stop_event = threading.Event()
    t_audio = threading.Thread(target=thread_receive_audio_from_mac, 
                               args=(args.mac_ip, args.port_audio_tx, stop_event))
    t_audio.daemon = True
    t_audio.start()
    resources["thread_audio_in"] = t_audio
    resources["stop_audio_thread_event"] = stop_event

    try:
        # 2. Start Outgoing Audio
        sock_audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_audio.connect((args.mac_ip, args.port_audio_rx))
        resources["sock_audio_out"] = sock_audio

        AudioCaptureModule = AudioStreamerModule("AudioCaptureModule", sock_audio)
        audio_proxy = ALProxy("ALAudioDevice", robot_ip, 9559)
        audio_proxy.setClientPreferences(AudioCaptureModule.getName(), 16000, 3, 0)
        audio_proxy.subscribe(AudioCaptureModule.getName())
        resources["audio_proxy"] = audio_proxy

        # 3. Start Outgoing Video
        sock_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_video.connect((args.mac_ip, args.port_video_rx))
        resources["sock_video"] = sock_video

        video_proxy = ALProxy("ALVideoDevice", robot_ip, 9559)
        name_id = video_proxy.subscribe("python_av_full", vision_definitions.kQVGA, vision_definitions.kBGRColorSpace, 15)
        video_proxy.setParam(vision_definitions.kCameraSelectID, args.camera_id)
        
        resources["video_proxy"] = video_proxy
        resources["video_name_id"] = name_id

        # 4. Configure Autonomous Life (Disable "Hey Nao", keep Tracking)
        try:
            life_proxy = ALProxy("ALAutonomousLife", robot_ip, 9559)
            # Disable the "Listening" ability to stop inherent keyword detection
            life_proxy.setAutonomousAbilityEnabled("Listening", False)
            # Ensure Basic Awareness (face/sound tracking) is enabled
            life_proxy.setAutonomousAbilityEnabled("BasicAwareness", True)
            resources["life_proxy"] = life_proxy
            print "[AutonomousLife] Specific keyword listening disabled; Tracking remains active."
        except Exception as e:
            print "[AutonomousLife] Warning: Could not configure capabilities: %s" % e

        print ">> SYSTEM STATUS: ONLINE (Streaming) <<"
    except Exception as e:
        print "Connection failed: %s" % e
        stop_services()
        raise e

def stop_services():
    global AudioCaptureModule, resources
    print "Stopping services..."
    
    if resources["stop_audio_thread_event"]: resources["stop_audio_thread_event"].set()
    if resources["audio_proxy"] and AudioCaptureModule:
        try: resources["audio_proxy"].unsubscribe(AudioCaptureModule.getName())
        except: pass
    if resources["video_proxy"] and resources["video_name_id"]:
        try: resources["video_proxy"].unsubscribe(resources["video_name_id"])
        except: pass
    
    # Restore Autonomous Life
    if resources["life_proxy"]:
        try:
            resources["life_proxy"].setAutonomousAbilityEnabled("Listening", True)
            print "[AutonomousLife] Listening restored."
        except: pass
        resources["life_proxy"] = None

    if resources["sock_audio_out"]:
        try: resources["sock_audio_out"].close()
        except: pass
        resources["sock_audio_out"] = None
    if resources["sock_video"]:
        try: resources["sock_video"].close()
        except: pass
        resources["sock_video"] = None
    print ">> SYSTEM STATUS: OFFLINE (Waiting) <<"

# --- MAIN ---

def main():
    global STATE_STREAMING
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--mac-ip", required=True)
    parser.add_argument("--port-audio-rx", type=int, default=50005)
    parser.add_argument("--port-video-rx", type=int, default=50006)
    parser.add_argument("--port-audio-tx", type=int, default=50007)
    parser.add_argument("--camera-id", type=int, default=0)
    args = parser.parse_args()

    try:
        myBroker = ALBroker("myBroker", "0.0.0.0", 0, args.robot_ip, 9559)
    except Exception as e:
        print "Error connecting to Naoqi: %s" % e
        sys.exit(1)
    
    global KeywordModule
    try:
        KeywordModule = KeywordService("KeywordModule")
        print "Waiting for keyword 'Start' or press ENTER to bypass..."
    except Exception as e:
        print "Error starting KeywordService: %s" % e
        myBroker.shutdown()
        sys.exit(1)

    is_connected = False

    try:
        while STATE_RUNNING:
            with mutex: target_streaming = STATE_STREAMING
            
            # Check for Enter key press without blocking the loop
            user_input = sys.stdin in select.select([sys.stdin], [], [], 0)[0]
            if user_input:
                sys.stdin.readline() # Clear the buffer
                with mutex: 
                    STATE_STREAMING = not STATE_STREAMING
                    print "\n>>> MANUAL TOGGLE: %s <<<" % ("STARTING" if STATE_STREAMING else "STOPPING")

            # 1. Handle Transitions
            if target_streaming and not is_connected:
                try:
                    start_services(args, args.robot_ip)
                    is_connected = True
                except:
                    with mutex: STATE_STREAMING = False

            elif not target_streaming and is_connected:
                stop_services()
                is_connected = False

            # 2. Handle Video Streaming
            if is_connected:
                try:
                    video_proxy = resources["video_proxy"]
                    name_id = resources["video_name_id"]
                    sock_video = resources["sock_video"]
                    al_img = video_proxy.getImageRemote(name_id)
                    if al_img and al_img[6]:
                        w, h = al_img[0], al_img[1]
                        raw = al_img[6]
                        # Reshape
                        frame = np.fromstring(raw, dtype=np.uint8).reshape((h, w, 3))
                        # Compress
                        res, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        if res:
                            d = enc.tostring()
                            sock_video.sendall(struct.pack(">L", len(d)) + d)
                except Exception: pass
                time.sleep(0.01)
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print "\nExit requested."
    finally:
        stop_services()
        if KeywordModule: KeywordModule.shutdown()
        myBroker.shutdown()

if __name__ == "__main__":
    main()