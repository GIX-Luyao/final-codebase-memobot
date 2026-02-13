# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script lets you to talk to a Gemini native audio model using the Live API.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Setup

To install the dependencies for this script, run:

```
brew install portaudio
pip install -U google-genai pyaudio
```

If Python < 3.11, also install `pip install taskgroup`.

## API key

Ensure the `GEMINI_API_KEY` environment variable is set to the api-key
you obtained from Google AI Studio.

## Run

To run the script:

```
python Get_started_LiveAPI_NativeAudio.py
```

Start talking to Gemini
"""

import asyncio
import json
import os
import sys
import traceback
from collections import deque

import pyaudio

from google import genai
from google.genai.types import (
    Content,
    LiveConnectConfig,
    Part,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    ExceptionGroup = exceptiongroup.ExceptionGroup
else:
    ExceptionGroup = ExceptionGroup

# Audio config (matches WebSocket reference: INPUT_RATE/OUTPUT_RATE/CHUNK)
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000   # Gemini SEND_SAMPLE_RATE
OUTPUT_RATE = 24000  # Gemini RECEIVE_SAMPLE_RATE
CHUNK = 512
VOICE_NAME = "Aoede"

pya = pyaudio.PyAudio()


class AudioManager:
    """Audio capture and playback; matches WebSocket client pattern (deque + async playback)."""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.in_stream = None
        self.out_stream = None
        self.play_queue = deque()
        self.playing_task = None
        self.is_running = True

    async def init(self):
        try:
            mic_info = self.p.get_default_input_device_info()
            self.in_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=INPUT_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK,
            )
            self.out_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=OUTPUT_RATE,
                output=True,
            )
            print("🎤 Mic + 🔈 Speaker initialized")
        except Exception as e:
            print(f"Error initializing audio: {e}")
            raise

    def read_mic(self):
        try:
            return self.in_stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading microphone: {e}")
            return b"\x00" * (CHUNK * 2)

    def queue_audio(self, data: bytes):
        if not self.is_running:
            return
        self.play_queue.append(data)
        if not self.playing_task or self.playing_task.done():
            self.playing_task = asyncio.create_task(self._playback())

    async def _playback(self):
        while self.play_queue and self.is_running:
            try:
                data = self.play_queue.popleft()
                await asyncio.to_thread(self.out_stream.write, data)
            except Exception as e:
                print(f"Error playing audio: {e}")
                break

    def interrupt(self, print_message=True):
        """Clear playback queue and cancel playing task. Set print_message=False for turn_complete."""
        self.play_queue.clear()
        if self.playing_task and not self.playing_task.done():
            self.playing_task.cancel()
        if print_message:
            print("🔇 Audio playback interrupted")

    def cleanup(self):
        self.is_running = False
        if self.in_stream:
            self.in_stream.close()
        if self.out_stream:
            self.out_stream.close()
        if self.p:
            self.p.terminate()
        print("🎤 Mic + 🔈 Speaker cleaned up")

# Check for API key before creating client
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
    print("\nMake sure you have either:", file=sys.stderr)
    print("  1. A .env file with: GOOGLE_API_KEY=your-api-key", file=sys.stderr)
    print("  2. Or export it: export GOOGLE_API_KEY='your-api-key'", file=sys.stderr)
    print("\nGet your API key from: https://aistudio.google.com/apikey", file=sys.stderr)
    sys.exit(1)

client = genai.Client(http_options={"api_version": "v1alpha"})

system_instruction_text = """
You are a helpful and friendly AI assistant.
Your default tone is helpful, engaging, and clear, with a touch of optimistic wit.
Anticipate user needs by clarifying ambiguous questions and always conclude your responses
with an engaging follow-up question to keep the conversation flowing.
"""

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
CONFIG = LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=VOICE_NAME)
        )
    ),
    system_instruction=Content(parts=[Part(text=system_instruction_text)]),
    tools=[],
)


def _config_with_transcription():
    """Config as dict with input/output transcription so API returns 'You said' / 'Gemini said'."""
    raw = CONFIG.model_dump(exclude_none=True)
    raw["input_audio_transcription"] = {}
    raw["output_audio_transcription"] = {}
    return raw


def _patch_live_setup_for_transcription():
    """Patch SDK so setup message includes input_audio_transcription (API requires it for transcriptions)."""
    try:
        from google.genai import live as live_module
        from google.genai import types as types_module
        _orig = live_module.AsyncLive._LiveSetup_to_mldev

        def _patched(self, model, config=None):
            result = _orig(self, model, config)
            from_object = config.model_dump(exclude_none=True) if isinstance(config, types_module.LiveConnectConfig) else (config or {})
            if from_object.get("input_audio_transcription") is not None and "setup" in result:
                result["setup"]["inputAudioTranscription"] = from_object["input_audio_transcription"]
            if from_object.get("output_audio_transcription") is not None and "setup" in result:
                result["setup"]["outputAudioTranscription"] = from_object["output_audio_transcription"]
            return result

        live_module.AsyncLive._LiveSetup_to_mldev = _patched
    except Exception:
        pass


def _patch_live_parse_transcription():
    """Patch SDK to pass through inputTranscription/outputTranscription from API into server_content."""
    try:
        from typing import Any, Optional
        from google.genai import live as live_module
        from google.genai import types as genai_types
        _orig_content_from_mldev = live_module.AsyncLive._LiveServerContent_from_mldev

        def _patched_content_from_mldev(self, from_object):
            to_object = _orig_content_from_mldev(self, from_object)
            sc = from_object if isinstance(from_object, dict) else getattr(from_object, "__dict__", {})
            if sc.get("inputTranscription") is not None:
                to_object["input_transcription"] = sc["inputTranscription"]
            if sc.get("outputTranscription") is not None:
                to_object["output_transcription"] = sc["outputTranscription"]
            return to_object

        live_module.AsyncLive._LiveServerContent_from_mldev = _patched_content_from_mldev

        _base = genai_types.LiveServerContent
        if not hasattr(_base, "input_transcription"):
            class _LiveServerContentWithTranscription(_base):
                input_transcription: Optional[Any] = None
                output_transcription: Optional[Any] = None
            genai_types.LiveServerContent = _LiveServerContentWithTranscription
    except Exception:
        pass


class AudioLoop:
    """Live API loop using AudioManager (same pattern as WebSocket client)."""
    def __init__(self):
        self.audio_mgr = AudioManager()
        self.out_queue = None
        self.session = None

    async def listen_audio(self):
        """Continuously capture mic and send to Gemini."""
        while self.audio_mgr.is_running:
            try:
                data = await asyncio.to_thread(self.audio_mgr.read_mic)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except Exception as e:
                print(f"Error in send_audio: {e}")
                break
            await asyncio.sleep(0.01)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def receive_audio(self):
        """Receive Gemini audio and transcriptions; match WebSocket client labels."""
        while self.audio_mgr.is_running:
            try:
                turn = self.session.receive()
                ai_text_started = False
                async for response in turn:
                    # Log all received payloads from websocket
                    log_parts = ["[WS recv]"]
                    if response.data:
                        log_parts.append(f"data={len(response.data)}b")
                    if response.text:
                        log_parts.append(f"text={repr(response.text)[:80]}")
                    sc = response.server_content
                    if sc is not None:
                        sc_dict = getattr(sc, "model_dump", lambda **kw: getattr(sc, "__dict__", {}))(
                            mode="json", exclude_none=True
                        ) if hasattr(sc, "model_dump") else getattr(sc, "__dict__", {})
                        sc_str = json.dumps(sc_dict, default=str) if isinstance(sc_dict, dict) else repr(sc_dict)
                        if len(sc_str) > 400:
                            sc_str = sc_str[:400] + "..."
                        log_parts.append(f"server_content={sc_str}")
                    if len(log_parts) > 1:
                        print(" ".join(log_parts))
                    if response.data:
                        self.audio_mgr.queue_audio(response.data)
                    if response.text:
                        if not ai_text_started:
                            print("\n🗣️ Gemini said: ", end="")
                            ai_text_started = True
                        print(response.text, end="")
                    sc = response.server_content
                    if sc:
                        if getattr(sc, "input_transcription", None):
                            it = sc.input_transcription
                            text = it.get("text", it) if isinstance(it, dict) else getattr(it, "text", it)
                            if text:
                                print(f"\n👤 You said: {text}")
                        if getattr(sc, "output_transcription", None):
                            ot = sc.output_transcription
                            text = ot.get("text", ot) if isinstance(ot, dict) else getattr(ot, "text", ot)
                            if text:
                                if not ai_text_started:
                                    print("\n🗣️ Gemini said: ", end="")
                                    ai_text_started = True
                                print(text, end="")
                        if sc.interrupted:
                            self.audio_mgr.interrupt(print_message=True)
                        elif sc.turn_complete:
                            if ai_text_started:
                                print()
                            self.audio_mgr.interrupt(print_message=False)
                            print("✅ Turn complete")
                self.audio_mgr.interrupt(print_message=False)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in recv_audio: {e}")
                break

    async def run(self):
        try:
            await self.audio_mgr.init()
            config = _config_with_transcription()
            _patch_live_setup_for_transcription()
            _patch_live_parse_transcription()
            async with (
                client.aio.live.connect(model=MODEL, config=config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.out_queue = asyncio.Queue(maxsize=5)
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
        except asyncio.CancelledError:
            pass
        except ExceptionGroup as eg:
            traceback.print_exception(eg)
        finally:
            self.audio_mgr.cleanup()


if __name__ == "__main__":
    loop = AudioLoop()
    asyncio.run(loop.run())