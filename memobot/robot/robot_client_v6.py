#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
import socket
import struct
import argparse
import threading
import subprocess
import tempfile
import math
import numpy as np
import cv2
import collections

from naoqi import ALProxy, ALBroker, ALModule
import vision_definitions

# --- GLOBALS ---
AudioCaptureModule = None
STATE_MODE = "WANDERING"  # Can be "WANDERING" or "INTERACTING"
STATE_RUNNING = True

wake_event = threading.Event()
sleep_event = threading.Event()

# Buffer for sound localization history
sound_buffer = collections.deque(maxlen=200)
sound_buffer_lock = threading.Lock()
WAKE_WORD_LAG = 1.0  # Seconds to look back for the sound source

# Resources
resources = {
    "sock_audio_out": None, 
    "sock_video": None, 
    "thread_audio_in": None,
    "thread_video_out": None,
    "audio_proxy": None, 
    "video_proxy": None, 
    "video_name_id": None,
    "life_proxy": None,
    "stop_event": threading.Event(),
}

# --- ALMODULE FOR MICROPHONE STREAMING ---

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


# --- BACKGROUND THREADS ---

def thread_receive_audio_from_mac(mac_ip, port, stop_event):
    """Receives TTS audio from the server and plays it via aplay."""
    print "[Audio RX] Connecting to Server Audio TX (Port %s)..." % port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect((mac_ip, port))
        print "[Audio RX] Connected."
    except Exception as e:
        print "[Audio RX] Failed to connect: %s" % e
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
        print "[Audio RX] Stopped."


def thread_sound_monitor(memory_proxy, motion_proxy, stop_event):
    """Continuously polls ALMemory for sound events to build a history buffer."""
    print "[SoundMonitor] Started tracking sound history."
    last_ts_seen = 0.0
    
    while not stop_event.is_set():
        try:
            # Poll ALAudioSourceLocalization/SoundLocated
            data = memory_proxy.getData("ALAudioSourceLocalization/SoundLocated")
            if data and len(data) > 1:
                # data[0] is [sec, usec]
                ts_sec = data[0][0]
                ts_usec = data[0][1]
                event_ts_robot = ts_sec + (ts_usec / 1000000.0)
                
                # Check if this is a new event frame
                if event_ts_robot != last_ts_seen:
                    last_ts_seen = event_ts_robot
                    azimuth = data[1][0]
                    energy = data[1][3]
                    
                    # Get current head yaw to compensate for head rotation
                    head_yaw = 0.0
                    try:
                        head_yaw = motion_proxy.getAngles("HeadYaw", True)[0]
                    except: pass

                    # Store with local system time for history lookup
                    with sound_buffer_lock:
                        sound_buffer.append({
                            'time': time.time(),
                            'azimuth': azimuth,
                            'head_yaw': head_yaw,
                            'energy': energy
                        })
        except Exception:
            pass
        time.sleep(0.05)
    print "[SoundMonitor] Stopped."


def thread_video_tx(mac_ip, port, robot_ip, camera_id, stop_event):
    """Continuously captures video frames and streams them to the server."""
    print "[Video TX] Connecting to Server Video RX (Port %s)..." % port
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock_video.connect((mac_ip, port))
        print "[Video TX] Connected."
    except Exception as e:
        print "[Video TX] Failed to connect: %s" % e
        return

    try:
        video_proxy = ALProxy("ALVideoDevice", robot_ip, 9559)
        name_id = video_proxy.subscribe("python_av_full", vision_definitions.kQVGA, vision_definitions.kBGRColorSpace, 15)
        video_proxy.setParam(vision_definitions.kCameraSelectID, camera_id)
        
        while not stop_event.is_set():
            al_img = video_proxy.getImageRemote(name_id)
            if al_img and al_img[6]:
                w, h = al_img[0], al_img[1]
                raw = al_img[6]
                frame = np.fromstring(raw, dtype=np.uint8).reshape((h, w, 3))
                res, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if res:
                    d = enc.tostring()
                    sock_video.sendall(struct.pack(">L", len(d)) + d)
            time.sleep(0.05)
    except Exception as e:
        print "[Video TX] Error: %s" % e
    finally:
        try: video_proxy.unsubscribe(name_id)
        except: pass
        sock_video.close()
        print "[Video TX] Stopped."


def thread_command_receiver(port, wake_event, sleep_event, stop_event):
    """Listens for direct commands (like WAKE) from the server."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(5)
    s.settimeout(1.0)
    print "[Cmd RX] Listening for 'WAKE' signals on port %d..." % port
    
    while not stop_event.is_set():
        try:
            conn, addr = s.accept()
            data = conn.recv(1024)
            if data:
                cmd = data.decode("utf-8").strip()
                if "WAKE" in cmd:
                    print "\n>>> COMMAND RECEIVED: WAKE <<<"
                    wake_event.set()
                elif "SLEEP" in cmd:
                    print "\n>>> COMMAND RECEIVED: SLEEP <<<"
                    sleep_event.set()
            conn.close()
        except socket.timeout: continue
        except Exception: pass
    s.close()
    print "[Cmd RX] Stopped."


def thread_code_receiver(mac_ip, port, stop_event):
    """Receives Python code from the server to execute dynamically."""
    def recv_exact(sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk: return None
            buf.extend(chunk)
        return bytes(buf)

    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((mac_ip, port))
            sock.settimeout(None)
            print "[Code RX] Connected to %s:%s" % (mac_ip, port)
            
            while not stop_event.is_set():
                header = recv_exact(sock, 4)
                if not header: break
                (msg_len,) = struct.unpack(">L", header)
                if msg_len <= 0 or msg_len > 10 * 1024 * 1024: break
                
                payload = recv_exact(sock, msg_len)
                if not payload: break
                
                code_str = payload.decode("utf-8", errors="replace")
                
                # Check if it's strictly a wake signal sent via code channel
                if code_str.strip() == "WAKE":
                    print "\n>>> CODE TRIGGER RECEIVED: WAKE <<<"
                    wake_event.set()
                elif code_str.strip() == "SLEEP":
                    print "\n>>> CODE TRIGGER RECEIVED: SLEEP <<<"
                    sleep_event.set()
                else:
                    print "\n" + "=" * 60 + "\nRECEIVED CODE FROM MAC\n" + "=" * 60
                    print code_str + "\n" + "=" * 60 + "\n"
                    run_received_code(code_str)
                    
        except Exception: pass
        finally:
            try: sock.close()
            except: pass
        if not stop_event.is_set(): time.sleep(2)


def run_received_code(code_str):
    try:
        fd, path = tempfile.mkstemp(suffix=".py", prefix="robot_code_")
        try:
            os.write(fd, code_str.encode("utf-8") if isinstance(code_str, type(u"")) else code_str)
            os.close(fd)
            fd = None
            proc = subprocess.Popen([sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out, _ = proc.communicate()
            if out: print "[Code STDOUT]\n%s" % (out.decode("utf-8", errors="replace").rstrip())
        finally:
            if fd is not None: os.close(fd)
            try: os.unlink(path)
            except: pass
    except Exception as e:
        print "[Code error] %s" % e


# --- HELPER FUNCTIONS ---

def stop_services():
    global AudioCaptureModule, resources
    print "Stopping services..."
    resources["stop_event"].set()
    
    if resources["audio_proxy"] and AudioCaptureModule:
        try: resources["audio_proxy"].unsubscribe(AudioCaptureModule.getName())
        except: pass
    
    if resources["life_proxy"]:
        try: resources["life_proxy"].setAutonomousAbilityEnabled("Listening", True)
        except: pass

    if resources["sock_audio_out"]:
        try: resources["sock_audio_out"].close()
        except: pass
    
    print ">> SYSTEM OFFLINE <<"

# --- MAIN LOOP ---

def main():
    global STATE_MODE, STATE_RUNNING, AudioCaptureModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--mac-ip", required=True)
    parser.add_argument("--port-audio-rx", type=int, default=50005) # Mac Mic Port
    parser.add_argument("--port-video-rx", type=int, default=50006) # Mac Video Port
    parser.add_argument("--port-audio-tx", type=int, default=50007) # Mac TTS Port
    parser.add_argument("--port-code-rx", type=int, default=50009)  # Mac Code Command Port
    parser.add_argument("--port-cmd-rx", type=int, default=50010)   # Nao's local listening port for WAKE
    parser.add_argument("--camera-id", type=int, default=0)
    args = parser.parse_args()

    # 1. Establish Naoqi Broker (Required for custom ALModules like audio capture)
    try:
        myBroker = ALBroker("myBroker", "0.0.0.0", 0, args.robot_ip, 9559)
    except Exception as e:
        print "Error connecting to Naoqi: %s" % e
        sys.exit(1)

    # 2. Setup Proxies
    try:
        motionProxy  = ALProxy("ALMotion", args.robot_ip, 9559)
        postureProxy = ALProxy("ALRobotPosture", args.robot_ip, 9559)
        sonarProxy   = ALProxy("ALSonar", args.robot_ip, 9559)
        memoryProxy  = ALProxy("ALMemory", args.robot_ip, 9559)
        audioLocProxy = ALProxy("ALAudioSourceLocalization", args.robot_ip, 9559)
        resources["life_proxy"] = ALProxy("ALAutonomousLife", args.robot_ip, 9559)
        
        # NAOqi 2.1: Disable Autonomous Life (no internal decisions) but manually start BasicAwareness
        # try:
        #     # 1. Disable Autonomous Life entirely
        #     if resources["life_proxy"].getState() != "disabled":
        #         resources["life_proxy"].setState("disabled")
            
        #     # 2. Manually start BasicAwareness for head tracking only
        #     awareness = ALProxy("ALBasicAwareness", args.robot_ip, 9559)
        #     # Ensure stimulus detection is on
        #     awareness.setStimulusDetectionEnabled("Sound", True)
        #     awareness.setStimulusDetectionEnabled("People", True)
        #     awareness.setParameter("LookStimulusSpeed", 0.7)
            
        #     # 3. Start it (independent of Autonomous Life)
        #     if not awareness.isAwarenessRunning():
        #         awareness.startAwareness()
        #     print "[BasicAwareness] Started manually (Sound+People tracking enabled)."
            
        # except Exception as e:
        #     print "Warning configuring AutonomousLife/Awareness: ", e

        # Init Localization (subscribe to populate ALMemory)
        audioLocProxy.setParameter("Sensitivity", 0.5)
        audioLocProxy.subscribe("WakeSoundLoc")
        
        # Init Sonar
        sonarProxy.subscribe("BlindWalker")
        l_sonar_key = "Device/SubDeviceList/US/Left/Sensor/Value"
        r_sonar_key = "Device/SubDeviceList/US/Right/Sensor/Value"

    except Exception as e:
        print "Could not create proxies: ", e

    print "Waking up..."
    motionProxy.wakeUp()
    postureProxy.goToPosture("StandInit", 0.5)

    # 3. Start Networking Streams
    stop_event = resources["stop_event"]
    
    # -> Outgoing Audio (Mic)
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_audio.connect((args.mac_ip, args.port_audio_rx))
    resources["sock_audio_out"] = sock_audio

    AudioCaptureModule = AudioStreamerModule("AudioCaptureModule", sock_audio)
    audio_proxy = ALProxy("ALAudioDevice", args.robot_ip, 9559)
    audio_proxy.setClientPreferences(AudioCaptureModule.getName(), 16000, 3, 0)
    audio_proxy.subscribe(AudioCaptureModule.getName())
    resources["audio_proxy"] = audio_proxy

    # -> Start background Threads
    t_audio_rx = threading.Thread(target=thread_receive_audio_from_mac, args=(args.mac_ip, args.port_audio_tx, stop_event))
    t_video_tx = threading.Thread(target=thread_video_tx, args=(args.mac_ip, args.port_video_rx, args.robot_ip, args.camera_id, stop_event))
    t_code_rx = threading.Thread(target=thread_code_receiver, args=(args.mac_ip, args.port_code_rx, stop_event))
    t_cmd_rx = threading.Thread(target=thread_command_receiver, args=(args.port_cmd_rx, wake_event, sleep_event, stop_event))
    t_sound_mon = threading.Thread(target=thread_sound_monitor, args=(memoryProxy, motionProxy, stop_event))
    
    for t in [t_audio_rx, t_video_tx, t_code_rx, t_cmd_rx, t_sound_mon]:
        t.daemon = True
        t.start()

    # --- WANDER CONFIG ---
    walking_speed = 0.5
    obs_threshold = 0.2
    turn_radians = 120 * (math.pi / 180.0)

    print ">> SYSTEM STATUS: ONLINE (Wandering) <<"
    print ">> Streaming audio and video to server. Waiting for Wake Word. <<"

    try:
        while STATE_RUNNING:
            if STATE_MODE == "WANDERING":
                # Check if Server detected the keyword
                if wake_event.is_set():
                    wake_event.clear()
                    STATE_MODE = "INTERACTING"
                    
                    # 1. Stop Moving
                    motionProxy.stopMove()
                    
                    # 2. Localize Sound and Turn
                    try:
                        # Attempt to find sound event from ~WAKE_WORD_LAG seconds ago
                        target_time = time.time() - WAKE_WORD_LAG
                        best_entry = None
                        min_diff = 1000.0
                        
                        with sound_buffer_lock:
                            for entry in sound_buffer:
                                diff = abs(entry['time'] - target_time)
                                if diff < min_diff:
                                    min_diff = diff
                                    best_entry = entry
                        
                        target_azimuth = None
                        
                        # Use historical data if it's reasonably close (e.g. within 1.0s of the target window)
                        if best_entry and min_diff < 1.0:
                            head_yaw = best_entry.get('head_yaw', 0.0)
                            target_azimuth = best_entry['azimuth'] + head_yaw
                            print ">> Using historical sound from %.2fs ago (diff=%.2fs, HeadAzimuth: %.2f, HeadYaw: %.2f, BodyAzimuth: %.2f)" % (WAKE_WORD_LAG, min_diff, best_entry['azimuth'], head_yaw, target_azimuth)
                        else:
                            print ">> Historical sound lookup failed (min_diff=%.2f). Fallback to immediate." % min_diff
                            # Fallback to immediate data
                            sound_data = memoryProxy.getData("ALAudioSourceLocalization/SoundLocated")
                            if sound_data and len(sound_data) > 1:
                                head_azimuth = sound_data[1][0]
                                head_yaw = 0.0
                                try:
                                    # Get current head yaw
                                    head_yaw = motionProxy.getAngles("HeadYaw", True)[0]
                                except: pass
                                target_azimuth = head_azimuth + head_yaw
                        
                        if target_azimuth is not None:
                            # Normalize angle to [-pi, pi] to take shortest turn
                            target_azimuth = (target_azimuth + np.pi) % (2 * np.pi) - np.pi
                            print "Turning to face speaker (Body Azimuth: %.2f rad)..." % target_azimuth
                            motionProxy.moveTo(0.0, 0.0, target_azimuth)
                            
                    except Exception as e:
                        print "Sound localization failed, skipping turn: ", e
                        
                    # # 3. Ready for interaction
                    # postureProxy.goToPosture("StandInit", 0.5)

                    # # Re-enable BasicAwareness to ensure head tracking during interaction
                    # try:
                    #     awareness = ALProxy("ALBasicAwareness", args.robot_ip, 9559)
                    #     awareness.setStimulusDetectionEnabled("Sound", True)
                    #     awareness.setStimulusDetectionEnabled("People", True)
                    #     awareness.setParameter("LookStimulusSpeed", 0.7)
                    #     if not awareness.isAwarenessRunning():
                    #         awareness.startAwareness()
                    #     print "[BasicAwareness] Resumed for interaction."
                    # except Exception as e:
                    #     print "Warning resuming BasicAwareness: ", e

                    print ">> ROBOT READY. (Interaction Active) <<"
                    
                else:
                    # Execute Wander Logic
                    val_left = memoryProxy.getData(l_sonar_key)
                    val_right = memoryProxy.getData(r_sonar_key)

                    if (val_left < obs_threshold) or (val_right < obs_threshold):
                        motionProxy.stopMove()
                        target_turn = -turn_radians if val_left < val_right else turn_radians
                        motionProxy.moveTo(0.0, 0.0, target_turn)
                    else:
                        motionProxy.move(walking_speed, 0.0, 0.0)
                        
                time.sleep(0.2)

            elif STATE_MODE == "INTERACTING":
                # Wait for optional reset command to go back to wandering
                if sleep_event.is_set():
                    sleep_event.clear()
                    STATE_MODE = "WANDERING"
                    print ">> RETURNING TO WANDER MODE <<"
                time.sleep(0.2)

    except KeyboardInterrupt:
        print "\nExit requested."
    finally:
        print "Cleaning up..."
        try:
            motionProxy.stopMove()
            sonarProxy.unsubscribe("BlindWalker")
            audioLocProxy.unsubscribe("WakeSoundLoc")
            motionProxy.rest()
        except: pass
        stop_services()
        myBroker.shutdown()

if __name__ == "__main__":
    main()