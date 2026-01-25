#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import base64
import json
import logging
import socket
import struct
import threading
import time
import wave
from collections import deque

import numpy as np
import cv2

from naoqi import ALProxy
import vision_definitions

try:
    from pocketsphinx import Decoder
except Exception:
    Decoder = None

# ---------- Hotword ----------
class HotwordKWS(object):
    def __init__(self, hotword="memobot", kws_threshold=1e-25):
        self.hotword = hotword.strip().lower()
        self.decoder = None
        self.last_hit = 0.0

        if Decoder is None:
            logging.warning("PocketSphinx not available; hotword disabled.")
            return

        # crude ARPAbet; tune as needed
        dict_text = "%s M EH M OW B AA T\n" % self.hotword.upper()

        import tempfile
        fd, dict_path = tempfile.mkstemp(prefix="memobot_", suffix=".dic")
        with open(dict_path, "w") as f:
            f.write(dict_text)

        cfg = Decoder.default_config()
        try:
            cfg.set_string('-logfn', '/dev/null')
        except Exception:
            pass
        cfg.set_string('-dict', dict_path)
        cfg.set_string('-keyphrase', self.hotword)
        cfg.set_float('-kws_threshold', float(kws_threshold))

        self.decoder = Decoder(cfg)
        self.decoder.start_utt()
        logging.info("Hotword enabled: %s", self.hotword)

    def feed(self, pcm16_bytes):
        if self.decoder is None:
            return False
        self.decoder.process_raw(pcm16_bytes, False, False)
        hyp = self.decoder.hyp()
        if hyp is not None:
            now = time.time()
            self.decoder.end_utt()
            self.decoder.start_utt()
            if now - self.last_hit > 1.0:
                self.last_hit = now
                return True
        return False

# ---------- Audio pull client ----------
def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

class PulledAudio(object):
    def __init__(self, robot_ip, audio_port, kws):
        self.robot_ip = robot_ip
        self.audio_port = int(audio_port)
        self.kws = kws
        self.latest = np.zeros((0,), dtype=np.int16)
        self.hotword_hit = False
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        th = threading.Thread(target=self._loop)
        th.daemon = True
        th.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((self.robot_ip, self.audio_port))
                s.settimeout(None)
                logging.info("Connected to robot audio stream %s:%d", self.robot_ip, self.audio_port)

                while not self._stop.is_set():
                    hdr = recvall(s, 4)
                    if hdr is None:
                        raise RuntimeError("audio stream closed")
                    nsamp = struct.unpack("!I", hdr)[0]
                    pcm = recvall(s, nsamp * 2)
                    if pcm is None:
                        raise RuntimeError("audio stream closed")

                    arr = np.fromstring(pcm, dtype=np.int16)
                    with self._lock:
                        self.latest = arr
                    if self.kws is not None and self.kws.feed(pcm):
                        self.hotword_hit = True
                        logging.info("HOTWORD DETECTED: MemoBot")

            except Exception as e:
                logging.warning("Audio stream error: %s", str(e))
                time.sleep(1.0)

    def get_latest(self):
        with self._lock:
            return self.latest.copy()

# ---------- Audio output to robot ----------
def read_wav_bytes(wav_bytes):
    try:
        from cStringIO import StringIO as BytesIO
    except Exception:
        from io import BytesIO
    bio = BytesIO(wav_bytes)
    wf = wave.open(bio, 'rb')
    ch = wf.getnchannels()
    sw = wf.getsampwidth()
    sr = wf.getframerate()
    n = wf.getnframes()
    pcm = wf.readframes(n)
    wf.close()
    if sw != 2:
        raise ValueError("WAV must be 16-bit PCM")
    samples = np.fromstring(pcm, dtype=np.int16)
    return samples, sr, ch

def to_stereo(samples, ch):
    if ch == 2:
        return samples
    mono = samples.reshape((-1,))
    stereo = np.empty((mono.size * 2,), dtype=np.int16)
    stereo[0::2] = mono
    stereo[1::2] = mono
    return stereo

class RobotAudioOut(object):
    def __init__(self, robot_ip, robot_port=9559):
        self.audio = ALProxy("ALAudioDevice", robot_ip, robot_port)
        try:
            self.audio.setParameter("outputSampleRate", 16000)
        except Exception:
            pass

    def play_wav_bytes(self, wav_bytes):
        samples, sr, ch = read_wav_bytes(wav_bytes)
        # For simplicity: require 16k wav; you can add resampling later if needed
        if sr != 16000:
            raise ValueError("Please send 16kHz WAV (got %d). Add resampling if you need it." % sr)
        stereo = to_stereo(samples, ch)
        frames_total = stereo.size // 2
        max_frames = 16384

        idx = 0
        while idx < frames_total:
            n = min(max_frames, frames_total - idx)
            chunk = stereo[(idx * 2):((idx + n) * 2)]
            ok = self.audio.sendRemoteBufferToOutput(int(n), chunk.tolist())
            if not ok:
                break
            time.sleep(float(n) / 16000.0)
            idx += n

# ---------- Command server (stdlib TCP JSON, one message per line) ----------
import SocketServer
class CmdHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            self.server.cmd_queue.append(obj)

class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

def start_cmd_server(host, port, cmd_queue):
    srv = ThreadedTCPServer((host, int(port)), CmdHandler)
    srv.cmd_queue = cmd_queue
    th = threading.Thread(target=srv.serve_forever)
    th.daemon = True
    th.start()
    logging.info("Command server listening on %s:%d (JSON lines over TCP)", host, int(port))
    return srv

# ---------- Plot ----------
def draw_wave(samples, w=640, h=240):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if samples is None or samples.size == 0:
        return img
    max_n = 1600
    if samples.size > max_n:
        samples = samples[-max_n:]
    y = samples.astype(np.float32) / 32768.0
    xs = np.linspace(0, w - 1, num=y.size).astype(np.int32)
    ys = (h / 2 - y * (h * 0.45)).astype(np.int32)
    pts = np.vstack((xs, ys)).T.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, (0, 255, 0), 1)
    cv2.line(img, (0, h // 2), (w - 1, h // 2), (80, 80, 80), 1)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", required=True)
    ap.add_argument("--robot-port", type=int, default=9559)
    ap.add_argument("--audio-port", type=int, default=20000, help="robot audio server port")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--hotword", default="memobot")
    ap.add_argument("--kws-threshold", type=float, default=1e-25)
    ap.add_argument("--cmd-host", default="0.0.0.0")
    ap.add_argument("--cmd-port", type=int, default=8765)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Video pull
    cam = ALProxy("ALVideoDevice", args.robot_ip, args.robot_port)
    name_id = cam.subscribe("memobot_cam", vision_definitions.kQVGA,
                            vision_definitions.kRGBColorSpace, args.fps)
    cam.setParam(vision_definitions.kCameraSelectID, args.camera_id)

    # Audio pull
    kws = HotwordKWS(args.hotword, args.kws_threshold)
    audio = PulledAudio(args.robot_ip, args.audio_port, kws)
    audio.start()

    # Robot audio out
    audio_out = RobotAudioOut(args.robot_ip, args.robot_port)

    # Command server
    cmd_queue = deque()
    start_cmd_server(args.cmd_host, args.cmd_port, cmd_queue)

    cv2.namedWindow("NAO Video", cv2.WINDOW_NORMAL)
    cv2.namedWindow("NAO Audio", cv2.WINDOW_NORMAL)

    try:
        while True:
            # video
            alimg = cam.getImageRemote(name_id)
            if alimg and len(alimg) >= 7:
                w, h = alimg[0], alimg[1]
                frame = np.fromstring(alimg[6], dtype=np.uint8).reshape((h, w, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("NAO Video", frame)

            # audio plot
            chunk = audio.get_latest()
            img = draw_wave(chunk)
            cv2.putText(img, "Hotword: %s" % args.hotword, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("NAO Audio", img)

            # handle commands
            while cmd_queue:
                obj = cmd_queue.popleft()
                t = obj.get("type", "")
                if t == "play_wav_b64":
                    wav_b64 = obj.get("wav_b64", "")
                    if wav_b64:
                        audio_out.play_wav_bytes(base64.b64decode(wav_b64))
                elif t == "say":
                    text = obj.get("text", "")
                    if text:
                        ALProxy("ALTextToSpeech", args.robot_ip, args.robot_port).say(text)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    finally:
        try:
            cam.unsubscribe(name_id)
        except Exception:
            pass
        audio.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
