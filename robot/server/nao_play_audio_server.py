# Save as mac_server_audio.py
import socket
import pyaudio
import sys

# Configuration
SERVER_IP = "0.0.0.0" # Listen on all interfaces
SERVER_PORT = 50005
CHUNK = 1024          # Buffer size
RATE = 16000          # Sample rate (matches NAO native)
FORMAT = pyaudio.paInt16
CHANNELS = 1

def main():
    # 1. Setup Audio Input (Mac Mic)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # 2. Setup TCP Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow socket reuse immediately after close
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(1)
        print("Mac Server listening on %s:%d..." % (SERVER_IP, SERVER_PORT))
        print("Waiting for NAO to connect...")
        
        client_socket, addr = server_socket.accept()
        print("NAO Connected from: %s" % str(addr))
        
        print("Streaming audio to NAO... (Ctrl+C to stop)")
        
        while True:
            # Read raw data from Mac Mic
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Send raw data to NAO
            client_socket.sendall(data)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print("Error: %s" % e)
    finally:
        print("Cleaning up...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        if 'client_socket' in locals(): client_socket.close()
        server_socket.close()

if __name__ == "__main__":
    main()
