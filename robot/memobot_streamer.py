#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import base64
import json
import logging
import threading
import time
import wave
from collections import deque

import numpy as np
import cv2

from naoqi import ALProxy, ALBroker, ALModule
import vision_definitions

try:
    from pocketsphinx import Decoder
except Exception:
    Decoder = None

try:
    from websocket_server import WebsocketServer
except Exception:
    WebsocketServer = None


# ALAudioDevice channel flags (commonly used in NAOqi examples):
# 0=ALL, 1=LEFT, 2=RIGHT, 3=FRONT, 4=REAR
ALL_CHANNELS = 0
LEFT_CHANNEL = 1
RIGHT_CHANNEL = 2
FRONT_CHANNEL = 3
REAR_CHANNEL = 4


def _safe_json_loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def _read_wav_bytes(wav_bytes):
    """
    Returns: (samples_int16, sample_rate, channels)
    samples_int16 is interleaved if channels>1.
    """
    # wave.open wants a file-like obj; in py2 use BytesIO from io
    try:
        from cStringIO import StringIO as BytesIO
    except Exception:
        from io import BytesIO

    bio = BytesIO(wav_bytes)
    wf = wave.open(bio, 'rb')
    channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    rate = wf.getframerate()
    nframes = wf.getnframes()
    pcm = wf.readframes(nframes)
    wf.close()

    if sampwidth != 2:
        raise ValueError("WAV must be 16-bit PCM (sampwidth=2). Got: %d" % sampwidth)

    samples = np.fromstring(pcm, dtype=np.int16)  # interleaved if stereo
    return samples, rate, channels


def _resample_linear_int16(samples_int16, src_rate, dst_rate, channels):
    """
    Simple linear resampler (good enough for voice TTS playback).
    samples_int16: interleaved if channels>1
    """
    if src_rate == dst_rate:
        return samples_int16

    if channels == 1:
        x = np.arange(len(samples_int16), dtype=np.float32)
        xp = np.linspace(0.0, float(len(samples_int16) - 1),
                         int(round(len(samples_int16) * float(dst_rate) / float(src_rate))),
                         dtype=np.float32)
        y = np.interp(xp, x, samples_int16.astype(np.float32))
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)

    # deinterleave
    frames = samples_int16.reshape((-1, channels))
    out_len = int(round(frames.shape[0] * float(dst_rate) / float(src_rate)))
    xp = np.linspace(0.0, float(frames.shape[0] - 1), out_len, dtype=np.float32)
    x = np.arange(frames.shape[0], dtype=np.float32)

    out = []
    for ch in range(channels):
        y = np.interp(xp, x, frames[:, ch].astype(np.float32))
        out.append(np.clip(np.round(y), -32768, 32767).astype(np.int16))
    out_frames = np.stack(out, axis=1)
    return out_frames.reshape((-1,)).astype(np.int16)


class HotwordKWS(object):
    """
    PocketSphinx keyword spotting for a single hotword (e.g., "memobot").
    """
    def __init__(self, hotword, kws_threshold=1e-25):
        self.hotword = hotword.strip().lower()
        self.kws_threshold = float(kws_threshold)
        self.decoder = None

        if Decoder is None:
            logging.warning("PocketSphinx not available; hotword spotting disabled.")
            return

        # Build a tiny dict containing the hotword.
        # You may need to tune the pronunciation for best results.
        # ARPAbet approx for "MEMOBOT": M EH M OW B AA T
        dict_text = "%s M EH M OW B AA T\n" % self.hotword.upper()

        # Create temp dict file
        import tempfile
        fd, dict_path = tempfile.mkstemp(prefix="memobot_dict_", suffix=".dic")
        with open(dict_path, "w") as f:
            f.write(dict_text)

        config = Decoder.default_config()
        # Silence logs (optional):
        try:
            config.set_string('-logfn', '/dev/null')
        except Exception:
            pass

        # Use default acoustic model shipped with pocketsphinx package
        # (usually auto-found by default_config).
        config.set_string('-dict', dict_path)

        config.set_string('-keyphrase', self.hotword)
        config.set_float('-kws_threshold', self.kws_threshold)

        self.decoder = Decoder(config)
        self.decoder.start_utt()

        logging.info("HotwordKWS enabled: hotword='%s', threshold=%g", self.hotword, self.kws_threshold)

    def feed(self, pcm16_bytes):
        """
        Feed 16kHz mono PCM16 (little-endian) bytes.
        Returns True if hotword detected in this chunk.
        """
        if self.decoder is None:
            return False
        self.decoder.process_raw(pcm16_bytes, False, False)
        hyp = self.decoder.hyp()
        if hyp is not None:
            # reset utterance for continuous spotting
            self.decoder.end_utt()
            self.decoder.start_utt()
            return True
        return False


class AudioReceiver(ALModule):
    """
    Receives NAO microphone buffers via ALAudioDevice and:
      - keeps a rolling buffer for display
      - runs hotword spotting
      - raises an ALMemory event on hotword
    """
    def __init__(self, name, robot_ip, robot_port, playback_flag, hotword="memobot", kws_threshold=1e-25):
        ALModule.__init__(self, name)

        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.playback_flag = playback_flag

        self.audio = ALProxy("ALAudioDevice", robot_ip, robot_port)
        self.mem = ALProxy("ALMemory", robot_ip, robot_port)

        self.sample_rate = 16000
        self.channel_flag = FRONT_CHANNEL
        self.deinterleave = 0

        self.kws = HotwordKWS(hotword=hotword, kws_threshold=kws_threshold)
        self.last_hotword_ts = 0.0

        self._lock = threading.Lock()
        self._last_chunk = np.zeros((0,), dtype=np.int16)
        self._ring = deque(maxlen=40)  # ~40 chunks (depends on NAO buffer size)

        self._running = False

    def start(self):
        if self._running:
            return
        # Request 1 channel @ 16kHz before subscribing. (Must be done before subscribe.) :contentReference[oaicite:4]{index=4}
        self.audio.setClientPreferences(self.getName(), self.sample_rate, self.channel_flag, self.deinterleave)
        self.audio.subscribe(self.getName())
        self._running = True
        logging.info("AudioReceiver subscribed (sr=%d, channel_flag=%d)", self.sample_rate, self.channel_flag)

    def stop(self):
        if not self._running:
            return
        try:
            self.audio.unsubscribe(self.getName())
        except Exception:
            pass
        self._running = False
        logging.info("AudioReceiver unsubscribed")

    def get_latest_chunk(self):
        with self._lock:
            return self._last_chunk.copy()

    # NAOqi will call this for remote audio clients
    def processRemote(self, nbOfChannels, nbrOfSamplesByChannel, aTimeStamp, buffer_audio):
        try:
            if buffer_audio is None:
                return

            # buffer_audio is bytes; each sample is int16
            interleaved = np.fromstring(str(buffer_audio), dtype=np.int16)

            if nbOfChannels > 1:
                # reshape by channels (Fortran order matches NAOqi examples)
                data = np.reshape(interleaved, (nbOfChannels, nbrOfSamplesByChannel), 'F')
                mono = data[0, :].copy()
            else:
                mono = interleaved.copy()

            with self._lock:
                self._last_chunk = mono
                self._ring.append(mono)

            # Hotword spotting: skip while we are playing robot audio to avoid feedback triggers
            if (self.kws is not None) and (not self.playback_flag.is_set()):
                if self.kws.feed(mono.tostring()):
                    now = time.time()
                    # simple debounce
                    if now - self.last_hotword_ts > 1.0:
                        self.last_hotword_ts = now
                        logging.info("HOTWORD DETECTED: MemoBot")
                        try:
                            self.mem.raiseEvent("MemoBot/Hotword", now)
                        except Exception as e:
                            logging.warning("Failed to raise ALMemory event: %s", str(e))

        except Exception as e:
            logging.warning("processRemote error: %s", str(e))


class RobotAudioOut(object):
    """
    Plays audio on NAO via ALAudioDevice.sendRemoteBufferToOutput
    (16-bit stereo interleaved; chunk size <= 16384 frames). :contentReference[oaicite:5]{index=5}
    """
    def __init__(self, robot_ip, robot_port, playback_flag):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.playback_flag = playback_flag
        self.audio = ALProxy("ALAudioDevice", robot_ip, robot_port)
        self.tts = None
        try:
            self.tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
        except Exception:
            pass

        # Set output sample rate to 16k for predictable playback
        try:
            self.audio.setParameter("outputSampleRate", 16000)
        except Exception:
            pass

        self._lock = threading.Lock()

    def _to_stereo_interleaved(self, samples_int16, channels):
        if channels == 2:
            return samples_int16
        if channels == 1:
            # duplicate mono to L/R
            mono = samples_int16.reshape((-1,))
            stereo = np.empty((mono.shape[0] * 2,), dtype=np.int16)
            stereo[0::2] = mono
            stereo[1::2] = mono
            return stereo
        # if >2, take first 2 channels
        frames = samples_int16.reshape((-1, channels))
        stereo2 = frames[:, :2].reshape((-1,)).astype(np.int16)
        return stereo2

    def play_wav_bytes(self, wav_bytes):
        with self._lock:
            self.playback_flag.set()
            try:
                if self.tts is not None:
                    # stop any ongoing TTS to reduce mixing
                    try:
                        self.tts.stopAll()
                    except Exception:
                        pass

                samples, rate, ch = _read_wav_bytes(wav_bytes)
                if rate != 16000:
                    samples = _resample_linear_int16(samples, rate, 16000, ch)
                    rate = 16000

                stereo = self._to_stereo_interleaved(samples, ch)
                # Send in chunks; each "frame" = 1 stereo sample pair = 2 int16 values
                frames_total = stereo.shape[0] // 2
                max_frames = 16384  # per NAOqi doc :contentReference[oaicite:6]{index=6}

                idx_frame = 0
                while idx_frame < frames_total:
                    n = min(max_frames, frames_total - idx_frame)
                    chunk = stereo[(idx_frame * 2):((idx_frame + n) * 2)]
                    # NAOqi expects ALValue-like list for remote buffer
                    ok = self.audio.sendRemoteBufferToOutput(int(n), chunk.tolist())
                    if not ok:
                        logging.warning("sendRemoteBufferToOutput returned False")
                        break
                    # pacing: sleep approximately chunk duration
                    time.sleep(float(n) / 16000.0)
                    idx_frame += n

            finally:
                # small cooldown to avoid immediate hotword re-trigger from tail audio
                time.sleep(0.2)
                self.playback_flag.clear()


class CommandServer(object):
    """
    WebSocket server: receives commands and pushes them to a queue.
    """
    def __init__(self, host, port, cmd_queue):
        if WebsocketServer is None:
            raise RuntimeError("websocket_server not installed. pip install websocket-server")
        self.host = host
        self.port = int(port)
        self.cmd_queue = cmd_queue
        self.server = WebsocketServer(host=self.host, port=self.port, loglevel=logging.INFO)

        self.server.set_fn_new_client(self._on_new_client)
        self.server.set_fn_client_left(self._on_client_left)
        self.server.set_fn_message_received(self._on_message)

    def _on_new_client(self, client, server):
        logging.info("WebSocket client connected: %s", str(client))

    def _on_client_left(self, client, server):
        logging.info("WebSocket client left: %s", str(client))

    def _on_message(self, client, server, message):
        obj = _safe_json_loads(message)
        if obj is None:
            logging.info("WS message (non-JSON): %s", message)
            return

        mtype = obj.get("type", "")
        if mtype == "play_wav_b64":
            wav_b64 = obj.get("wav_b64", "")
            if not wav_b64:
                return
            try:
                wav_bytes = base64.b64decode(wav_b64)
                self.cmd_queue.append(("play_wav_bytes", wav_bytes))
                logging.info("Queued play_wav_b64 (%d bytes)", len(wav_bytes))
            except Exception as e:
                logging.warning("Bad wav_b64: %s", str(e))

        elif mtype == "play_wav_url":
            url = obj.get("url", "")
            if not url:
                return
            self.cmd_queue.append(("play_wav_url", url))
            logging.info("Queued play_wav_url: %s", url)

        elif mtype == "say":
            text = obj.get("text", "")
            self.cmd_queue.append(("say", text))

        else:
            logging.info("Unknown WS type: %s", mtype)

    def run_forever(self):
        logging.info("WebSocket server listening on ws://%s:%d", self.host, self.port)
        self.server.run_forever()


def draw_waveform_window(samples_int16, width=640, height=240):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if samples_int16 is None or samples_int16.size == 0:
        return img

    # pick last ~0.1s if too large
    max_n = 1600
    if samples_int16.size > max_n:
        samples_int16 = samples_int16[-max_n:]

    # normalize to [-1,1]
    y = samples_int16.astype(np.float32) / 32768.0

    xs = np.linspace(0, width - 1, num=y.size).astype(np.int32)
    ys = (height / 2 - y * (height * 0.45)).astype(np.int32)

    pts = np.vstack((xs, ys)).T.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, (0, 255, 0), 1)
    cv2.line(img, (0, height // 2), (width - 1, height // 2), (80, 80, 80), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", required=True, help="NAO robot IP (e.g., 192.168.1.10)")
    ap.add_argument("--robot-port", type=int, default=9559, help="NAOqi port (default 9559)")
    ap.add_argument("--broker-ip", default="0.0.0.0", help="Local broker bind IP (default 0.0.0.0)")
    ap.add_argument("--broker-port", type=int, default=0, help="Local broker bind port (0=random)")
    ap.add_argument("--camera-id", type=int, default=0, help="0=top, 1=bottom (default 0)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--hotword", default="memobot")
    ap.add_argument("--kws-threshold", type=float, default=1e-25)
    ap.add_argument("--ws-host", default="0.0.0.0")
    ap.add_argument("--ws-port", type=int, default=8765)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    robot_ip = args.robot_ip
    robot_port = args.robot_port

    playback_flag = threading.Event()

    # Start NAOqi broker (needed for remote audio callbacks)
    broker = ALBroker("memobot_broker", args.broker_ip, args.broker_port, robot_ip, robot_port)

    # Audio receiver module
    audio_receiver = AudioReceiver(
        "AudioReceiver",
        robot_ip=robot_ip,
        robot_port=robot_port,
        playback_flag=playback_flag,
        hotword=args.hotword,
        kws_threshold=args.kws_threshold
    )
    audio_receiver.start()

    # Audio output helper
    audio_out = RobotAudioOut(robot_ip, robot_port, playback_flag)

    # Video setup
    cam = ALProxy("ALVideoDevice", robot_ip, robot_port)
    resolution = vision_definitions.kQVGA
    colorSpace = vision_definitions.kRGBColorSpace
    name_id = cam.subscribe("memobot_cam", resolution, colorSpace, args.fps)
    cam.setParam(vision_definitions.kCameraSelectID, args.camera_id)

    # Command queue and worker
    cmd_queue = deque()
    stop_flag = threading.Event()

    def cmd_worker():
        import requests
        tts = None
        try:
            tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
        except Exception:
            pass

        while not stop_flag.is_set():
            if not cmd_queue:
                time.sleep(0.01)
                continue

            cmd, payload = cmd_queue.popleft()

            try:
                if cmd == "play_wav_bytes":
                    audio_out.play_wav_bytes(payload)

                elif cmd == "play_wav_url":
                    r = requests.get(payload, timeout=10)
                    r.raise_for_status()
                    audio_out.play_wav_bytes(r.content)

                elif cmd == "say":
                    if tts is not None and payload:
                        tts.say(payload)

            except Exception as e:
                logging.warning("Command failed (%s): %s", cmd, str(e))

    worker_th = threading.Thread(target=cmd_worker)
    worker_th.daemon = True
    worker_th.start()

    # WebSocket server thread
    if WebsocketServer is not None:
        ws = CommandServer(args.ws_host, args.ws_port, cmd_queue)
        ws_th = threading.Thread(target=ws.run_forever)
        ws_th.daemon = True
        ws_th.start()
    else:
        logging.warning("websocket_server not installed; WS control disabled.")

    logging.info("Streaming started. Press 'q' to quit.")
    cv2.namedWindow("NAO Video", cv2.WINDOW_NORMAL)
    cv2.namedWindow("NAO Audio", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Video frame
            alimg = cam.getImageRemote(name_id)
            if alimg is not None and len(alimg) >= 7:
                w = alimg[0]
                h = alimg[1]
                array = alimg[6]
                frame = np.fromstring(array, dtype=np.uint8).reshape((h, w, 3))
                # alimg is RGB; OpenCV expects BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if playback_flag.is_set():
                    cv2.putText(frame_bgr, "PLAYBACK", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("NAO Video", frame_bgr)

            # Audio waveform window
            chunk = audio_receiver.get_latest_chunk()
            wav_img = draw_waveform_window(chunk)
            cv2.putText(wav_img, "Hotword: %s" % args.hotword, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("NAO Audio", wav_img)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    finally:
        stop_flag.set()
        try:
            cam.unsubscribe(name_id)
        except Exception:
            pass
        audio_receiver.stop()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()
