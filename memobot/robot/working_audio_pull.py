import sys
import time
from naoqi import ALProxy, ALBroker, ALModule

AudioModule = None

class AudioPrinterModule(ALModule):
    def __init__(self, name):
        ALModule.__init__(self, name)

    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, timestamp, buffer):
        # buffer is a python2 'str' of raw 16-bit PCM
        data_preview = buffer[:10].encode('hex')
        print "Channels: %d | Samples/Ch: %d | Preview: %s..." % (
            nbOfChannels, nbrOfSamplesByChannel, data_preview
        )

def main(ip, port=9559):
    # Broker makes your module reachable by the robot
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, ip, port)

    global AudioModule
    AudioModule = AudioPrinterModule("AudioModule")

    audio_device = ALProxy("ALAudioDevice", ip, port)

    # Use setClientPreferences (NOT setClientContext)
    # name, sampleRate, channels, deinterleaved
    # For single channel, NAOqi supports 16000Hz. :contentReference[oaicite:2]{index=2}
    audio_device.setClientPreferences(AudioModule.getName(), 16000, 3, 0)  # 3 = front mic :contentReference[oaicite:3]{index=3}

    audio_device.subscribe(AudioModule.getName())

    print "Streaming started. Ctrl+C to stop."
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print "Stopping..."
        audio_device.unsubscribe(AudioModule.getName())
        myBroker.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python jason_audio_stream.py <ROBOT_IP>"
    else:
        main(sys.argv[1])
