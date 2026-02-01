#!/usr/bin/env python3
"""
Mock TCP client that streams Mac microphone and camera to mac_master_v4.
Use this to test mac_master_v4 without a physical robot: it acts as the "robot"
sending audio (mic) and video (webcam) to the server.
"""

import argparse
import socket
import struct
import sys
import threading
import time

import cv2
import pyaudio

# --- CONFIG (must match mac_master_v3 / mac_master_v4 server) ---
DEFAULT_HOST = "127.0.0.1"
PORT_AUDIO_RX = 50005
PORT_VIDEO_RX = 50006
PORT_AUDIO_TX = 50007  # server sends audio TO robot → we play on default speaker

# Audio: match server expectation (robot / NAO style)
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 4096

# Video
JPEG_QUALITY = 50
VIDEO_FPS_TARGET = 20


def run_audio_client(host: str, port: int, stop_event: threading.Event) -> None:
    """Stream Mac microphone to server (audio RX)."""
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
    except Exception as e:
        print(f"[Audio] Failed to open microphone: {e}")
        return

    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)
            print(f"[Audio] Connected to {host}:{port}")
            try:
                while not stop_event.is_set():
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    if not data:
                        break
                    sock.sendall(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                sock.close()
        except (ConnectionRefusedError, OSError) as e:
            print(f"[Audio] Connection failed: {e}. Retrying in 2s...")
        if not stop_event.is_set():
            time.sleep(2)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("[Audio] Stopped.")


def run_video_client(host: str, port: int, stop_event: threading.Event) -> None:
    """Stream Mac camera to server (video RX). Protocol: 4-byte big-endian length + JPEG bytes."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Video] Failed to open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    frame_interval = 1.0 / VIDEO_FPS_TARGET

    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)
            print(f"[Video] Connected to {host}:{port}")
            next_frame_time = time.monotonic()
            try:
                while not stop_event.is_set():
                    now = time.monotonic()
                    if now < next_frame_time:
                        time.sleep(min(0.01, next_frame_time - now))
                        continue
                    next_frame_time = now + frame_interval

                    ret, frame = cap.read()
                    if not ret:
                        break
                    result, encimg = cv2.imencode(".jpg", frame, encode_param)
                    if not result:
                        continue
                    data = encimg.tobytes()
                    msg_size = len(data)
                    sock.sendall(struct.pack(">L", msg_size) + data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                sock.close()
        except (ConnectionRefusedError, OSError) as e:
            print(f"[Video] Connection failed: {e}. Retrying in 2s...")
        if not stop_event.is_set():
            time.sleep(2)

    cap.release()
    print("[Video] Stopped.")


def list_output_devices(p: pyaudio.PyAudio) -> None:
    """Print available output devices so user can pick one (e.g. AirPods)."""
    print("[Speaker] Available output devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxOutputChannels", 0) > 0:
            name = info.get("name", "?")
            print(f"  [{i}] {name}")


def get_first_output_device_index(p: pyaudio.PyAudio) -> int | None:
    """Return the first available output device index (fallback when default is invalid)."""
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxOutputChannels", 0) > 0:
            return i
    return None


def find_output_device_index(p: pyaudio.PyAudio, device: str | None) -> int | None:
    """Resolve device to an index: int string -> index; otherwise search by name (e.g. 'AirPods')."""
    if not (device or "").strip():
        return None
    device = device.strip()
    if device.isdigit():
        return int(device)
    device_lower = device.lower()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxOutputChannels", 0) > 0 and device_lower in (info.get("name") or "").lower():
            return i
    return None


def run_audio_receiver(
    host: str,
    port: int,
    stop_event: threading.Event,
    output_device_index: int | None = None,
) -> None:
    """Connect to server audio TX and play incoming audio on the given output device (or default)."""
    p = pyaudio.PyAudio()
    stream = None
    device_desc = f"device {output_device_index}" if output_device_index is not None else "default speaker"

    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)
            print(f"[Speaker] Connected to {host}:{port} (incoming audio -> {device_desc})")
            if stream is None:
                kwargs = dict(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=CHUNK,
                )
                if output_device_index is not None:
                    kwargs["output_device_index"] = output_device_index
                try:
                    stream = p.open(**kwargs)
                except (OSError, Exception) as open_err:
                    err_str = str(open_err)
                    if output_device_index is None and ("Invalid output device" in err_str or "-9996" in err_str):
                        fallback = get_first_output_device_index(p)
                        if fallback is not None:
                            kwargs["output_device_index"] = fallback
                            stream = p.open(**kwargs)
                            name = p.get_device_info_by_index(fallback).get("name", "?")
                            device_desc = f"fallback: {name}"
                            print(f"[Speaker] Default device invalid; using first available output: {name}")
                        else:
                            raise
                    else:
                        raise
            try:
                while not stop_event.is_set():
                    data = sock.recv(CHUNK)
                    if not data:
                        break
                    stream.write(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                sock.close()
        except (ConnectionRefusedError, OSError) as e:
            err_str = str(e)
            if "Invalid output device" in err_str or "-9996" in err_str:
                print(f"[Speaker] Output device error: {e}")
                print("[Speaker] Use --list-output-devices to see devices, then --output-device 'AirPods' or --output-device <index>")
                list_output_devices(p)
            else:
                print(f"[Speaker] Connection failed: {e}. Retrying in 2s...")
        except Exception as e:
            err_str = str(e)
            if "Invalid output device" in err_str or "-9996" in err_str:
                print(f"[Speaker] Output device error: {e}")
                print("[Speaker] Use --list-output-devices to see devices, then --output-device 'AirPods' or --output-device <index>")
                list_output_devices(p)
            else:
                print(f"[Speaker] Connection failed: {e}. Retrying in 2s...")
        if not stop_event.is_set():
            time.sleep(2)

    if stream:
        stream.stop_stream()
        stream.close()
    p.terminate()
    print("[Speaker] Stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Mock TCP client: stream Mac mic + camera to mac_master_v4."
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"mac_master_v4 server host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--audio-port",
        type=int,
        default=PORT_AUDIO_RX,
        help=f"Server port for audio RX (default: {PORT_AUDIO_RX})",
    )
    parser.add_argument(
        "--video-port",
        type=int,
        default=PORT_VIDEO_RX,
        help=f"Server port for video RX (default: {PORT_VIDEO_RX})",
    )
    parser.add_argument(
        "--audio-tx-port",
        type=int,
        default=PORT_AUDIO_TX,
        help=f"Server port for audio TX / speaker playback (default: {PORT_AUDIO_TX})",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=None,
        metavar="NAME_OR_INDEX",
        help="Output device for speaker (e.g. 'AirPods' or device index). Use --list-output-devices to see options.",
    )
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help="List available output devices and exit (use with --output-device to pick one).",
    )
    args = parser.parse_args()

    if args.list_output_devices:
        p = pyaudio.PyAudio()
        list_output_devices(p)
        p.terminate()
        return

    output_device_index = None
    if args.output_device:
        p = pyaudio.PyAudio()
        output_device_index = find_output_device_index(p, args.output_device)
        p.terminate()
        if output_device_index is None:
            print(f"[Speaker] No output device matching '{args.output_device}'. Use --list-output-devices to see options.")
            sys.exit(1)
        print(f"[Speaker] Using output device: {args.output_device} (index {output_device_index})")

    stop = threading.Event()
    threads = [
        threading.Thread(target=run_audio_client, args=(args.host, args.audio_port, stop), daemon=True),
        threading.Thread(target=run_video_client, args=(args.host, args.video_port, stop), daemon=True),
        threading.Thread(
            target=run_audio_receiver,
            args=(args.host, args.audio_tx_port, stop),
            kwargs={"output_device_index": output_device_index},
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    out_desc = f"output device '{args.output_device}'" if args.output_device else "default speaker"
    print(f"Mock client running (mic + camera -> server; server audio -> {out_desc}). Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    stop.set()
    for t in threads:
        t.join(timeout=3)
    print("Done.")


if __name__ == "__main__":
    main()
