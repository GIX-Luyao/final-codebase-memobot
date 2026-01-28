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

    # Video setup
    cam = ALProxy("ALVideoDevice", robot_ip, robot_port)
    resolution = vision_definitions.kQVGA
    colorSpace = vision_definitions.kRGBColorSpace
    name_id = cam.subscribe("memobot_cam", resolution, colorSpace, args.fps)
    cam.setParam(vision_definitions.kCameraSelectID, args.camera_id)

    # Command queue and worker
    cmd_queue = deque()
    stop_flag = threading.Event()


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

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    finally:
        stop_flag.set()
        try:
            cam.unsubscribe(name_id)
        except Exception:
            pass
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()
