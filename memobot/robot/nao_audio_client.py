import sys
import time
import socket
from naoqi import ALProxy, ALBroker, ALModule

# TCP Configuration
SERVER_IP = “10.19.171.37”  # Replace with your Mac's IP address
SERVER_PORT = 50005

AudioModule = None

class AudioStreamerModule(ALModule):
    def __init__(self, name, client_socket):
        ALModule.__init__(self, name)
        self.client_socket = client_socket

    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, timestamp, buffer):
        """
        buffer is a python2 'str' containing raw 16-bit PCM data.
        We send the raw bytes directly over the TCP socket.
        """
        try:
            self.client_socket.sendall(buffer)
        except Exception as e:
            print "Socket error: %s" % e

def main(robot_ip, robot_port=9559):
    # 1. Connect to the Mac Server first
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print "Connected to Mac server at %s" % SERVER_IP
    except Exception as e:
        print "Could not connect to server: %s" % e
        return

    # 2. Setup NAOqi Broker
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, robot_ip, robot_port)

    global AudioModule
    AudioModule = AudioStreamerModule("AudioModule", client_socket)

    audio_device = ALProxy("ALAudioDevice", robot_ip, robot_port)

    # NAO v4 defaults: 16000Hz, 3 (Front Mic), 0 (Interleaved)
    audio_device.setClientPreferences(AudioModule.getName(), 16000, 3, 0)
    audio_device.subscribe(AudioModule.getName())

    print "Streaming to Mac started. Ctrl+C to stop."
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print "Stopping..."
        audio_device.unsubscribe(AudioModule.getName())
        myBroker.shutdown()
        client_socket.close()
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python stream_to_mac.py <ROBOT_IP>"
    else:
        main(sys.argv[1])
