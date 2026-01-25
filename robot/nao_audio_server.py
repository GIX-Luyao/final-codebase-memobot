import socket
import pyaudio

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 50005
CHANNELS = 1      # NAO provides 1 channel if configured as '3' (Front mic)
RATE = 16000      # Must match the NAOqi client preference
FORMAT = pyaudio.paInt16

def start_server():
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)

    # Initialize Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"Server listening on {HOST}:{PORT}...")
    print("Waiting for NAO to connect...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    try:
        while True:
            # Receive PCM bytes (adjust buffer size if audio crackles)
            data = conn.recv(4096)
            if not data:
                break
            # Play the bytes through speakers
            stream.write(data)
    except KeyboardInterrupt:
        print("\nStopping Server...")
    finally:
        conn.close()
        server_socket.close()
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    start_server()