#!/usr/bin/env python3
import socket
import struct
import threading
import os
import time
import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CONFIGURATION ---
ROBOT_IP = "10.19.57.226"  # <--- CHECK YOUR ROBOT IP
PORT_VIDEO = 50006
PORT_CMD   = 50007
MODEL_ID   = "gemini-2.0-flash"

# --- COMMANDS ---
CMD_STOP       = 0
CMD_FORWARD    = 1
CMD_LEFT       = 2
CMD_RIGHT      = 3
CMD_BACKWARD   = 4 # Used for "STUCK"
CMD_LOOK_LEFT  = 5
CMD_LOOK_RIGHT = 6
CMD_ALIGN_BODY = 7

CMD_NAMES = {
    0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT", 
    4: "STUCK", 5: "HEAD_L", 6: "HEAD_R", 7: "ALIGNING"
}

# --- SHARED STATE ---
latest_frame = None
current_command = CMD_STOP
frame_lock = threading.Lock()
command_lock = threading.Lock()
ai_paused = False # Flag to pause normal AI loop during escape maneuver

# --- GEMINI SETUP ---
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found.")
    exit()

client = genai.Client(api_key=api_key)

def get_latest_jpeg():
    """Helper to get current frame as bytes"""
    with frame_lock:
        if latest_frame is not None:
            _, buf = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            return buf.tobytes()
    return None

def send_force_command(cmd_id):
    """Updates global command immediately"""
    global current_command
    with command_lock:
        current_command = cmd_id

def analyze_scene(prompt_text, img_bytes):
    """Generic Gemini Caller"""
    if not img_bytes: return "UNKNOWN"
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part(text=prompt_text),
                types.Part(inline_data={"mime_type": "image/jpeg", "data": img_bytes})
            ]
        )
        return response.text.strip().upper()
    except Exception as e:
        print(f"AI Error: {e}")
        return "ERROR"

def escape_routine():
    """
    Complex maneuver: Look around -> Decide -> Turn Body -> Reset
    """
    global ai_paused
    print("⚠️ OBSTACLE DETECTED! Starting Escape Maneuver...")
    ai_paused = True # Stop the main AI loop from interfering
    
    try:
        # 1. STOP
        send_force_command(CMD_STOP)
        time.sleep(1.0)
        
        # 2. LOOK LEFT
        print("👀 Looking LEFT...")
        send_force_command(CMD_LOOK_LEFT)
        time.sleep(2.0) # Wait for head to move
        left_view = get_latest_jpeg()
        
        # 3. LOOK RIGHT
        print("👀 Looking RIGHT...")
        send_force_command(CMD_LOOK_RIGHT)
        time.sleep(2.0) # Wait for head to move
        right_view = get_latest_jpeg()
        
        # 4. DECIDE
        # We ask Gemini to compare or judge the views.
        # Simple approach: Ask specifically about the view currently in front
        prompt_check = "Is this path clear enough to walk? YES or NO."
        
        print("🧠 Analyzing LEFT view...")
        left_res = analyze_scene(prompt_check, left_view)
        
        print("🧠 Analyzing RIGHT view...")
        right_res = analyze_scene(prompt_check, right_view)
        
        decision = CMD_STOP
        
        # Prefer the one that says YES. If both YES, Left. If both NO, Turn Around (Right x2)
        if "YES" in left_res:
            print("✅ LEFT path selected.")
            send_force_command(CMD_LOOK_LEFT) # Move head back to target
            time.sleep(1.0)
            decision = CMD_ALIGN_BODY
        elif "YES" in right_res:
            print("✅ RIGHT path selected.")
            send_force_command(CMD_LOOK_RIGHT) # Move head back to target
            time.sleep(1.0)
            decision = CMD_ALIGN_BODY
        else:
            print("❌ Both blocked. Defaulting to full turn.")
            send_force_command(CMD_LOOK_LEFT) # Just pick one to unstick
            time.sleep(1.0)
            decision = CMD_ALIGN_BODY

        # 5. EXECUTE BODY TURN
        if decision == CMD_ALIGN_BODY:
            print("🔄 Turning BODY to match HEAD...")
            send_force_command(CMD_ALIGN_BODY)
            time.sleep(4.0) # Allow time for physical turn
            
    finally:
        print("▶️ Resuming Normal AI Drive.")
        send_force_command(CMD_STOP)
        ai_paused = False

def ai_loop():
    """Main Driving Logic"""
    global current_command, ai_paused
    print("🧠 AI Brain initialized (BOLD MODE).")
    
    # Updated Prompt for Aggressive Driving
    system_instruction = """
    You are controlling a robot. Be BOLD.
    1. If there is ANY gap slightly wider than you, go FORWARD.
    2. Do NOT stop for small objects on the floor, walk over them.
    3. Only turn LEFT or RIGHT if the path is completely blocked.
    4. If you are facing a flat wall or corner with NO way out, output STUCK.
    
    OUTPUT one word: FORWARD, LEFT, RIGHT, STUCK.
    """
    
    while True:
        if ai_paused: 
            time.sleep(0.5)
            continue
            
        img_bytes = get_latest_jpeg()
        
        if img_bytes:
            text = analyze_scene(system_instruction, img_bytes)
            
            if "STUCK" in text:
                # Trigger the complex sequence in a separate thread so we don't block video
                threading.Thread(target=escape_routine).start()
                time.sleep(5) # Dead time to let the routine take over
            else:
                new_cmd = CMD_STOP
                if "FORWARD" in text: new_cmd = CMD_FORWARD
                elif "LEFT" in text: new_cmd = CMD_LEFT
                elif "RIGHT" in text: new_cmd = CMD_RIGHT
                
                with command_lock:
                    current_command = new_cmd
            
            time.sleep(0.3) # Fast reaction
        else:
            time.sleep(0.1)

def video_client_thread():
    """Connects to Robot and PULLS video."""
    global latest_frame
    print(f"📷 Connecting to Robot Video ({ROBOT_IP}:{PORT_VIDEO})...")
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((ROBOT_IP, PORT_VIDEO))
            print("✅ Video Connected!")
            while True:
                raw_len = s.recv(4)
                if not raw_len: break
                msg_len = struct.unpack(">L", raw_len)[0]
                data = b''
                while len(data) < msg_len:
                    packet = s.recv(msg_len - len(data))
                    if not packet: break
                    data += packet
                if len(data) == msg_len:
                    nparr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    with frame_lock:
                        latest_frame = frame
        except Exception:
            time.sleep(3)

def command_client_thread():
    """Connects to Robot and PUSHES commands."""
    print(f"🎮 Connecting to Robot Command ({ROBOT_IP}:{PORT_CMD})...")
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((ROBOT_IP, PORT_CMD))
            print("✅ Command Link Established!")
            while True:
                with command_lock:
                    cmd_to_send = current_command
                s.send(struct.pack("B", cmd_to_send))
                time.sleep(0.1) # 10Hz command rate
        except Exception:
            time.sleep(3)

if __name__ == "__main__":
    t1 = threading.Thread(target=video_client_thread, daemon=True)
    t2 = threading.Thread(target=command_client_thread, daemon=True)
    t3 = threading.Thread(target=ai_loop, daemon=True)
    t1.start(); t2.start(); t3.start()
    
    print("🖥️  PC Controller Running...")
    
    while True:
        display_frame = None
        cmd_text = "..."
        with frame_lock:
            if latest_frame is not None: display_frame = latest_frame.copy()
        with command_lock:
            cmd_text = CMD_NAMES.get(current_command, "UNKNOWN")
            
        if display_frame is not None:
            cv2.rectangle(display_frame, (0, 0), (320, 50), (0,0,0), -1)
            cv2.putText(display_frame, f"AI: {cmd_text}", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Robot View", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()