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
        print "Mac Server listening on %s:%d..." % (SERVER_IP, SERVER_PORT)
        print "Waiting for NAO to connect..."
        
        client_socket, addr = server_socket.accept()
        print "NAO Connected from: %s" % str(addr)
        
        print "Streaming audio to NAO... (Ctrl+C to stop)"
        
        while True:
            # Read raw data from Mac Mic
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Send raw data to NAO
            client_socket.sendall(data)

    except KeyboardInterrupt:
        print "\nStopping..."
    except Exception as e:
        print "Error: %s" % e
    finally:
        print "Cleaning up..."
        stream.stop_stream()
        stream.close()
        p.terminate()
        if 'client_socket' in locals(): client_socket.close()
        server_socket.close()

if __name__ == "__main__":
    main()






# Save as stream_from_mac.py
import sys
import socket
import subprocess
from naoqi import ALProxy

# Configuration
SERVER_IP = "10.19.171.37" # Replace with your Mac's IP
SERVER_PORT = 50005
BUFFER_SIZE = 4096

def main(robot_ip, robot_port=9559):
    # 1. Optional: Set Volume using NAOqi
    try:
        audio_device = ALProxy("ALAudioDevice", robot_ip, robot_port)
        audio_device.setOutputVolume(70) # Set volume to 70%
        print "Volume set to 70%"
    except Exception as e:
        print "Warning: Could not set volume via ALAudioDevice: %s" % e

    # 2. Connect to Mac Server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print "Connecting to Mac at %s:%s..." % (SERVER_IP, SERVER_PORT)
        s.connect((SERVER_IP, SERVER_PORT))
        print "Connected!"
    except Exception as e:
        print "Connection failed: %s" % e
        return

    # 3. Setup 'aplay' subprocess
    # We pipe data directly to stdin. 
    # -t raw: Raw data (no header)
    # -r 16000: Sample rate
    # -f S16_LE: Signed 16-bit Little Endian
    # -c 1: Mono
    cmd = ['aplay', '-t', 'raw', '-r', '16000', '-f', 'S16_LE', '-c', '1']
    
    try:
        # Launch the audio player process
        audio_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        print "Streaming started. Ctrl+C to stop."
        
        while True:
            # Receive data from TCP
            data = s.recv(BUFFER_SIZE)
            if not data:
                break
            
            # Write data to the audio player's standard input
            audio_process.stdin.write(data)
            
    except KeyboardInterrupt:
        print "\nStopping stream..."
    except Exception as e:
        print "Error during streaming: %s" % e
    finally:
        s.close()
        if 'audio_process' in locals():
            audio_process.stdin.close()
            audio_process.terminate()
            print "Audio process terminated."

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python stream_from_mac.py <ROBOT_IP>"
    else:
        main(sys.argv[1])