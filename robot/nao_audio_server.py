#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import time
import socket
import struct
import array
import threading

from naoqi import ALProxy, ALModule, ALBroker

# ----------------------------
# Simple TCP broadcast server
# ----------------------------
class TcpBroadcaster(object):
    """
    Laptop connects to robot:PORT. Robot pushes frames:
      [4-byte big-endian uint32 = samples_per_channel] + [raw PCM bytes int16 little-endian interleaved]
    """
    def __init__(self, bind_ip, bind_port):
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self._srv = None
        self._clients = []
        self._lock = threading.Lock()

    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.bind_ip, self.bind_port))
        self._srv.listen(5)
        t = threading.Thread(target=self._accept_loop)
        t.daemon = True
        t.start()
        print("[AUDIO] TCP server listening on %s:%d" % (self.bind_ip, self.bind_port))

    def _accept_loop(self):
        while True:
            c, addr = self._srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._clients.append(c)
            print("[AUDIO] client connected:", addr)

    def send_frame(self, samples_per_channel, pcm_bytes):
        if not pcm_bytes:
            return
        header = struct.pack("!I", int(samples_per_channel))
        payload = header + pcm_bytes

        dead = []
        with self._lock:
            for c in self._clients:
                try:
                    c.sendall(payload)
                except Exception:
                    dead.append(c)
            for c in dead:
                try:
                    c.close()
                except Exception:
                    pass
                if c in self._clients:
                    self._clients.remove(c)

# --------------------------------
# NAOqi Audio callback module
# --------------------------------
class RobotAudioStreamer(ALModule):
    def __init__(self, name, nao_ip, nao_port, out_ip, out_port,
                 sample_rate=16000, channel_cfg=3, deinterleaved=0):
        """
        channel_cfg: for NAOqi single-channel configs, many examples use:
          0=ALL, 1=LEFT, 2=RIGHT, 3=FRONT, 4=REAR (varies by doc/version).
        """
        ALModule.__init__(self, name)

        # IMPORTANT: export callbacks so NAOqi can call them
        # (this pattern is required for NAOqi to invoke Python callbacks) :contentReference[oaicite:1]{index=1}
        try:
            self.BIND_PYTHON(self.getName(), "process")
        except Exception:
            pass
        try:
            self.BIND_PYTHON(self.getName(), "processRemote")
        except Exception:
            pass
        try:
            self.BIND_PYTHON(self.getName(), "processSoundRemote")
        except Exception:
            pass

        self.nao_ip = nao_ip
        self.nao_port = nao_port
        self.sample_rate = sample_rate
        self.channel_cfg = channel_cfg
        self.deinterleaved = deinterleaved

        self.audio = ALProxy("ALAudioDevice", nao_ip, nao_port)
        self.bcast = TcpBroadcaster(out_ip, out_port)
        self.bcast.start()

        self._first_cb = True

    def start(self):
        # If inputs were closed by something else, reopen them.
        try:
            if hasattr(self.audio, "isInputClosed") and self.audio.isInputClosed():
                self.audio.openAudioInputs()
        except Exception:
            pass

        # Must match the module name you subscribe with :contentReference[oaicite:2]{index=2}
        self.audio.setClientPreferences(self.getName(),
                                        int(self.sample_rate),
                                        int(self.channel_cfg),
                                        int(self.deinterleaved))

        # Try subscribe() first (newer style), fallback to subscribeRemoteModule() (older style)
        ok = None
        try:
            ok = self.audio.subscribe(self.getName())
            print("[AUDIO] subscribe(%s) => %s" % (self.getName(), str(ok)))
        except Exception as e:
            print("[AUDIO] subscribe() failed:", repr(e))

        if not ok:
            try:
                ok2 = self.audio.subscribeRemoteModule(self.getName())
                print("[AUDIO] subscribeRemoteModule(%s) => %s" % (self.getName(), str(ok2)))
            except Exception as e:
                print("[AUDIO] subscribeRemoteModule() failed:", repr(e))

        print("[AUDIO] waiting for callbacks... (you should see 'FIRST AUDIO CALLBACK' within 1s)")

    def stop(self):
        try:
            self.audio.unsubscribe(self.getName())
        except Exception:
            pass
        try:
            self.audio.unSubscribeRemoteModule(self.getName())
        except Exception:
            pass

    # ---- buffer conversion helpers ----
    def _to_pcm_bytes(self, buf):
        """
        Goal: raw PCM int16 little-endian bytes.
        buf can be:
          - str / bytearray (already bytes)
          - list of 0..255 (bytes)
          - list of int16 samples
        """
        if buf is None:
            return ""

        # Python2: bytes are 'str'
        if isinstance(buf, str):
            return buf
        if isinstance(buf, bytearray):
            return str(buf)

        if isinstance(buf, list):
            if len(buf) == 0:
                return ""
            mn = min(buf)
            mx = max(buf)
            # looks like bytes
            if mn >= 0 and mx <= 255:
                a = array.array('B', buf)
                return a.tostring()
            # looks like int16 samples
            a = array.array('h', buf)
            return a.tostring()

        # fallback
        try:
            return str(buf)
        except Exception:
            return ""

    def _handle_audio(self, cb_name, nb_ch, nb_samp, ts, buf):
        pcm = self._to_pcm_bytes(buf)

        if self._first_cb:
            self._first_cb = False
            print("[AUDIO] FIRST AUDIO CALLBACK via %s: nb_ch=%s nb_samp=%s len(pcm)=%d ts=%s"
                  % (cb_name, str(nb_ch), str(nb_samp), len(pcm), str(ts)))

        # Push to laptop
        self.bcast.send_frame(nb_samp, pcm)

    # -------------------------
    # Possible callback names
    # -------------------------

    # Newer docs: subscribe() calls process(...) :contentReference[oaicite:3]{index=3}
    def process(self, nbOfChannels, nbrOfSamplesByChannel, a, b):
        # Some versions swap (buffer, timestamp) vs (timestamp, buffer).
        # Heuristic: timestamp is usually a list/tuple of 2 numbers.
        ts = None
        buf = None

        def looks_like_ts(x):
            return isinstance(x, (list, tuple)) and len(x) == 2

        if looks_like_ts(a) and not looks_like_ts(b):
            ts, buf = a, b
        elif looks_like_ts(b) and not looks_like_ts(a):
            ts, buf = b, a
        else:
            # assume doc order: buffer then timestamp :contentReference[oaicite:4]{index=4}
            buf, ts = a, b

        self._handle_audio("process", nbOfChannels, nbrOfSamplesByChannel, ts, buf)

    # Common Python example pattern: processRemote(nbCh, nbSamples, timeStamp, inputBuffer)
    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        self._handle_audio("processRemote", nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer)

    # Older/alt docs: processSoundRemote(nbCh, nbSamples, dataInterleaved) :contentReference[oaicite:5]{index=5}
    def processSoundRemote(self, nbOfChannels, nbOfSamplesByChannel, dataInterleaved):
        self._handle_audio("processSoundRemote", nbOfChannels, nbOfSamplesByChannel, None, dataInterleaved)


def main():
    if len(sys.argv) < 5:
        print("Usage: python robot_audio_streamer.py <nao_ip> <nao_port> <laptop_ip> <laptop_port>")
        sys.exit(1)

    nao_ip = sys.argv[1]
    nao_port = int(sys.argv[2])
    laptop_ip = sys.argv[3]
    laptop_port = int(sys.argv[4])

    # IMPORTANT: When running on the robot itself, bind broker to 127.0.0.1 to avoid “wrong interface” callback issues.
    broker = ALBroker("RobotAudioStreamerBroker",
                      "127.0.0.1", 0,
                      "127.0.0.1", nao_port)

    mod = RobotAudioStreamer("RobotAudioStreamer",
                             nao_ip=nao_ip, nao_port=nao_port,
                             out_ip="0.0.0.0", out_port=laptop_port,  # listen for laptop on robot
                             sample_rate=16000, channel_cfg=3, deinterleaved=0)

    mod.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        mod.stop()
        broker.shutdown()

if __name__ == "__main__":
    main()
