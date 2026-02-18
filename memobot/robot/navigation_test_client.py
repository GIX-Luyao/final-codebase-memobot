#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import struct
import threading
import time
import sys
import numpy as np
import cv2
import math
from naoqi import ALProxy, ALBroker

# --- CONFIGURATION ---
PORT_VIDEO = 50006
PORT_CMD   = 50007

# --- COMMANDS ---
CMD_STOP       = 0
CMD_FORWARD    = 1
CMD_LEFT       = 2
CMD_RIGHT      = 3
CMD_BACKWARD   = 4
CMD_LOOK_LEFT  = 5
CMD_LOOK_RIGHT = 6
CMD_ALIGN_BODY = 7

# --- GLOBAL STATE ---
OBSTACLE_DETECTED = False
stop_event = threading.Event()

def fix_head_position(robot_ip):
    """Initial Head Setup"""
    print "[Init] 🔒 Locking Head Position..."
    try:
        life = ALProxy("ALAutonomousLife", robot_ip, 9559)
        if life.getState() != "disabled":
            life.setState("disabled")
        
        posture = ALProxy("ALRobotPosture", robot_ip, 9559)
        posture.goToPosture("StandInit", 0.5)
        
        motion = ALProxy("ALMotion", robot_ip, 9559)
        motion.setStiffnesses("Head", 1.0)
        motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 0.2)
        print "[Init] ✅ Head Fixed Forward."
    except Exception as e:
        print "[Init] ⚠️ Init Warning: %s" % e

def safety_thread(memory):
    """
    Reads Sonar. We only stop for IMMEDIATE collision.
    The 'Bold' logic is handled by the PC, but this prevents physical crashes.
    """
    global OBSTACLE_DETECTED
    print "[Safety] Sonar Monitor active."
    while not stop_event.is_set():
        try:
            l = memory.getData("Device/SubDeviceList/US/Left/Sensor/Value")
            r = memory.getData("Device/SubDeviceList/US/Right/Sensor/Value")
            # Only trigger if extremely close (< 30cm)
            if l < 0.3 or r < 0.3:
                OBSTACLE_DETECTED = True
            else:
                OBSTACLE_DETECTED = False
        except:
            pass
        time.sleep(0.1)

def video_server_thread(video_proxy):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", PORT_VIDEO))
    server_sock.listen(1)
    
    # 0=TopCamera, 11=RGB, 10fps
    sub_id = video_proxy.subscribe("NavCam", 0, 11, 10)
    print "[Video] 📷 Listening on Port %d..." % PORT_VIDEO
    
    while not stop_event.is_set():
        client, addr = server_sock.accept()
        print "[Video] ✅ Connected: %s" % str(addr)
        
        try:
            while not stop_event.is_set():
                img = video_proxy.getImageRemote(sub_id)
                if img:
                    width, height = img[0], img[1]
                    raw_data = img[6]
                    nparr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                    bgr_img = cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)
                    ret, jpeg = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    
                    if ret:
                        data = jpeg.tostring()
                        header = struct.pack(">L", len(data))
                        client.sendall(header + data)
                time.sleep(0.05)
        except:
            pass
        finally:
            client.close()

def command_server_thread(motion):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", PORT_CMD))
    server_sock.listen(1)
    
    print "[Command] 🎮 Listening on Port %d..." % PORT_CMD
    
    while not stop_event.is_set():
        client, addr = server_sock.accept()
        print "[Command] ✅ Connected: %s" % str(addr)
        
        try:
            while not stop_event.is_set():
                data = client.recv(1)
                if not data: break
                cmd = struct.unpack("B", data)[0]
                
                # --- EXECUTION LOGIC ---
                
                # 1. Safety Override (Only overrides Forward)
                if OBSTACLE_DETECTED and cmd == CMD_FORWARD:
                    motion.stopMove()
                    continue
                
                # 2. Locomotion
                if cmd == CMD_FORWARD:
                    # Faster walk (Bold)
                    motion.move(0.2, 0, 0)
                    # Keep head straight while walking
                    motion.setAngles("HeadYaw", 0.0, 0.1)
                    
                elif cmd == CMD_LEFT:
                    motion.move(0, 0, 0.3)
                elif cmd == CMD_RIGHT:
                    motion.move(0, 0, -0.3)
                elif cmd == CMD_STOP:
                    motion.stopMove()
                    
                # 3. Head Turning (Scanning)
                elif cmd == CMD_LOOK_LEFT:
                    motion.stopMove()
                    # Look 45 degrees left (approx 0.8 rad)
                    motion.setAngles("HeadYaw", 0.8, 0.2)
                    
                elif cmd == CMD_LOOK_RIGHT:
                    motion.stopMove()
                    # Look 45 degrees right
                    motion.setAngles("HeadYaw", -0.8, 0.2)
                    
                # 4. Body Alignment (The "Fix Head, Turn Body" Move)
                elif cmd == CMD_ALIGN_BODY:
                    motion.stopMove()
                    
                    # Get current Head Angle
                    current_yaw = motion.getAngles("HeadYaw", True)[0]
                    print "[Action] Aligning Body to Head: %f rad" % current_yaw
                    
                    # Holonomic turn (x=0, y=0, theta=current_yaw)
                    # This turns the body so the head faces front (relative to body)
                    motion.moveTo(0, 0, current_yaw)
                    
                    # Reset Head to center immediately
                    motion.setAngles("HeadYaw", 0.0, 0.2)
                    
        except Exception as e:
            print "[Command] Error: %s" % e
        finally:
            client.close()
            motion.stopMove()

def main():
    try:
        broker = ALBroker("myBroker", "0.0.0.0", 0, "127.0.0.1", 9559)
    except:
        sys.exit(1)

    motion = ALProxy("ALMotion")
    video  = ALProxy("ALVideoDevice")
    memory = ALProxy("ALMemory")
    sonar  = ALProxy("ALSonar")
    
    motion.wakeUp()
    sonar.subscribe("SafetyApp")
    
    fix_head_position("127.0.0.1")
    
    t1 = threading.Thread(target=video_server_thread, args=(video,))
    t2 = threading.Thread(target=command_server_thread, args=(motion,))
    t3 = threading.Thread(target=safety_thread, args=(memory,))
    
    t1.start(); t2.start(); t3.start()
    
    print "✅ Robot Ready."
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        motion.stopMove()
        sonar.unsubscribe("SafetyApp")

if __name__ == "__main__":
    main()