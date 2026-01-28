robot



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Save this as robot_video_streamer.py on the NAO robot

import socket
import struct
import time
import argparse
import numpy as np
import cv2
from naoqi import ALProxy
import vision_definitions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac-ip", required=True, help="IP address of the Mac server")
    parser.add_argument("--mac-port", type=int, default=50006, help="Port to connect to on Mac")
    parser.add_argument("--camera-id", type=int, default=0, help="0=Top, 1=Bottom")
    args = parser.parse_args()

    # 1. Connect to Local Camera (NAOqi)
    # Using '127.0.0.1' because this script runs ON the robot
    print "Connecting to local Naoqi Camera proxy..."
    video_proxy = ALProxy("ALVideoDevice", "127.0.0.1", 9559)
    
    # Resolution: 320x240 (QVGA) is best for streaming speed
    resolution = vision_definitions.kQVGA
    color_space = vision_definitions.kBGRColorSpace # OpenCV native format
    fps = 15
    
    name_id = video_proxy.subscribe("python_streamer", resolution, color_space, fps)
    video_proxy.setParam(vision_definitions.kCameraSelectID, args.camera_id)
    print "Camera subscribed: " + name_id

    # 2. Connect to Mac TCP Server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print "Connecting to Mac at %s:%d..." % (args.mac_ip, args.mac_port)
        client_socket.connect((args.mac_ip, args.mac_port))
        print "Connected!"

        while True:
            # A. Capture Frame
            al_img = video_proxy.getImageRemote(name_id)
            
            if al_img and al_img[6]:
                # Extract image data
                width = al_img[0]
                height = al_img[1]
                raw_data = al_img[6]
                
                # Convert raw buffer to numpy array
                # Using fromstring for compatibility with older numpy versions
                frame = np.fromstring(raw_data, dtype=np.uint8).reshape((height, width, 3))

                # B. Compress Frame (JPEG)
                # This reduces data size by ~90%, essential for smooth streaming
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                result, encimg = cv2.imencode('.jpg', frame, encode_param)
                
                if result:
                    # FIX: Use tostring() instead of tobytes() for older Numpy versions
                    data = encimg.tostring()
                    size = len(data)
                    
                    # C. Send Frame: [4-byte Size] + [Image Bytes]
                    # Pack size as unsigned long (network byte order)
                    client_socket.sendall(struct.pack(">L", size) + data)

            # Optional: Short sleep to maintain loop stability
            time.sleep(0.01)

    except KeyboardInterrupt:
        print "\nStopping..."
    except Exception as e:
        print "Error: %s" % str(e)
    finally:
        print "Cleaning up..."
        try:
            video_proxy.unsubscribe(name_id)
        except:
            pass
        client_socket.close()

if __name__ == "__main__":
    main()



—-----------------------------------------------------------------------------------------------------------

Server 



# Save as mac_video_server.py
import socket
import struct
import cv2
import numpy as np
import sys

# Configuration
SERVER_IP = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 50006    # Different port than your audio server (50005)

def recv_exact(sock, n):
    """Helper to receive exactly n bytes."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(1)
        print("Mac Video Server listening on {}:{}...".format(SERVER_IP, SERVER_PORT))
        print("Waiting for Robot to connect...")
        
        conn, addr = server_socket.accept()
        print("Robot Connected from: {}".format(addr))
        
        payload_size = struct.calcsize(">L")
        
        while True:
            # 1. Read Message Size (4 bytes)
            packed_msg_size = recv_exact(conn, payload_size)
            if not packed_msg_size:
                break
            
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            
            # 2. Read Image Data (msg_size bytes)
            frame_data = recv_exact(conn, msg_size)
            if not frame_data:
                break

            # 3. Decode JPEG to Frame
            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            # 4. Display
            if frame is not None:
                cv2.imshow("NAO Robot Stream", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print("Error: {}".format(e))
    finally:
        if 'conn' in locals(): conn.close()
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
