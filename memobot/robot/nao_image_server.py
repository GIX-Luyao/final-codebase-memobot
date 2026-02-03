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
