import sys
import time
import socket
import threading
from naoqi import ALProxy, ALBroker, ALModule

SERVER_IP = "10.19.171.37"
SERVER_PORT = 50005

AudioModule = None
is_playing_audio = threading.Event()

class AudioStreamerModule(ALModule):
    def __init__(self, name, client_socket):
        ALModule.__init__(self, name)
        self.client_socket = client_socket

    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, timestamp, buffer):
        if not is_playing_audio.is_set():
            try:
                self.client_socket.sendall(buffer)
            except:
                pass

def receive_and_play(client_socket, player_proxy):
    """
    On NAOqi 2.1, we use a different approach. 
    Instead of playRaw, we will use a small workaround.
    """
    try:
        while True:
            data = client_socket.recv(4096)
            if not data:
                break
            
            is_playing_audio.set()
            
            # Since playRaw failed, we use the ALAudioDevice's internal 
            # 'sendRemoteBufferToOutput' if available, or post a task.
            # But the most reliable 2.1 method for raw bytes is actually:
            try:
                # We try the direct device write
                audio_device = ALProxy("ALAudioDevice", player_proxy.getBrokerName().split(":")[0], 9559)
                audio_device.sendRemoteBufferToOutput(len(data)/2, data)
            except:
                pass
            
            # Use a short sleep to allow the buffer to clear before resuming mic
            is_playing_audio.clear()
    except Exception as e:
        print "Receiver error: %s" % e

def main(robot_ip, robot_port=9559):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
    except Exception as e:
        print "Connection failed: %s" % e
        return

    myBroker = ALBroker("myBroker", "0.0.0.0", 0, robot_ip, robot_port)
    
    global AudioModule
    AudioModule = AudioStreamerModule("AudioModule", client_socket)
    
    # We use ALAudioDevice for the mic and the low-level output buffer
    audio_device = ALProxy("ALAudioDevice", robot_ip, robot_port)

    receiver_thread = threading.Thread(target=receive_and_play, args=(client_socket, audio_device))
    receiver_thread.daemon = True
    receiver_thread.start()

    audio_device.setClientPreferences(AudioModule.getName(), 16000, 3, 0)
    audio_device.subscribe(AudioModule.getName())

    print "Bidirectional Stream active (NAOqi 2.1 Fix). Ctrl+C to stop."
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio_device.unsubscribe(AudioModule.getName())
        myBroker.shutdown()
        client_socket.close()

if __name__ == "__main__":
    main(sys.argv[1])
