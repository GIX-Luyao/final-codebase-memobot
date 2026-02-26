#!/usr/bin/env python3
import sys
import unittest
import numpy as np
import time

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    FuncAnimation = None
    MATPLOTLIB_AVAILABLE = False

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    pvporcupine = None
    PORCUPINE_AVAILABLE = False
import os
import struct
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

class TestMicrophoneInput(unittest.TestCase):
    
    def setUp(self):
        # Load .env file
        if load_dotenv:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                print(f"[Test] Loading .env from {env_path}")
                load_dotenv(dotenv_path=env_path)
            else:
                 print(f"[Test] .env file not found at {env_path}")
        else:
            print("[Test] python-dotenv not installed. Skipping .env load.")

        if not SOUNDDEVICE_AVAILABLE:
            self.skipTest("sounddevice not installed. Install with: pip install sounddevice")
        
        # Audio configuration matching mac_master_v10.py
        self.ROBOT_AUDIO_RATE = 16000
        self.MIC_BLOCKSIZE = 2048 # samples
        self.stream = None

    def tearDown(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
    # ... (rest of the file)

    def _try_open_stream(self, device_idx=None):
        s = sd.InputStream(
            samplerate=self.ROBOT_AUDIO_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.MIC_BLOCKSIZE,
            device=device_idx,
        )
        s.start()
        return s

    def get_working_stream(self):
        """Returns an open stream using the fallback logic."""
        mic_stream = None
        device_used = "Default"

        # 1. First Priority: Default Device
        try:
            mic_stream = self._try_open_stream(None)
            print("[Test] ✅ Success: Opened system microphone (default device).")
        except Exception as e:
            print(f"[Test] ⚠️ Default mic failed: {e}. Attempting fallback search...")
            
            # 2. Fallback: Search for valid inputs
            try:
                devices = sd.query_devices()
                candidates = []
                for i, d in enumerate(devices):
                    if d['max_input_channels'] > 0:
                        name = d.get('name', '').lower()
                        score = 0
                        # Heuristics for likely working mics
                        if 'usb' in name: score += 10
                        if 'default' in name: score += 5
                        if 'pulse' in name: score += 4
                        if 'sysdefault' in name: score += 3
                        
                        candidates.append((score, i, d.get('name', 'Unknown')))
                
                # Sort by score desc
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                for score, idx, name in candidates:
                    try:
                        print(f"[Test] Trying fallback device {idx} ('{name}')...")
                        mic_stream = self._try_open_stream(idx)
                        device_used = f"Device {idx} ({name})"
                        print(f"[Test] ✅ Success! Using fallback device: {device_used}")
                        break
                    except Exception as ex:
                        print(f"[Test] Device {idx} failed: {ex}")
            except Exception as e2:
                print(f"[Test] Device enumeration failed: {e2}")

        if mic_stream is None:
            self.fail("Failed to open any microphone input stream.")
            
        self.stream = mic_stream
        return mic_stream, device_used

    def test_microphone_plot(self):
        """
        Runs a real-time plot of the microphone input, AND attempts to detect wake word if Porcupine is available.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[Test] Matplotlib not found. Install it to see the plot: pip install matplotlib")
            # Fallback to text-based VU meter if no matplotlib
            self._run_text_vu_meter()
            return

        print("\n[Test] Starting Real-Time Audio Plot + Wake Word Detection...")
        print("[Test] Close the plot window to stop the test.")
        
        # Setup Porcupine
        porcupine = None
        porcupine_buffer = bytearray()
        porcupine_frame_bytes = 0
        porcupine_status_text = "Porcupine: OFF"
        
        if PORCUPINE_AVAILABLE:
            access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
            if access_key:
                try:
                    porcupine = pvporcupine.create(access_key=access_key, keywords=["jarvis"])
                    porcupine_frame_bytes = porcupine.frame_length * 2
                    porcupine_status_text = "Porcupine: LISTENING (Say 'Jarvis')"
                    print(f"[Test] Porcupine initialized. Frame length: {porcupine.frame_length}")
                except Exception as e:
                    print(f"[Test] Porcupine init failed: {e}")
                    porcupine_status_text = f"Porcupine Error: {e}"
            else:
                print("[Test] PICOVOICE_ACCESS_KEY not set. Wake word detection disabled.")
                porcupine_status_text = "Porcupine: NO KEY"

        stream, device_name = self.get_working_stream()
        
        # Setup Plot
        fig, ax = plt.subplots()
        x = np.arange(0, self.MIC_BLOCKSIZE)
        line, = ax.plot(x, np.zeros(self.MIC_BLOCKSIZE))
        
        ax.set_ylim(-32768, 32767)
        ax.set_xlim(0, self.MIC_BLOCKSIZE)
        ax.set_title(f"Microphone Input: {device_name}\n{porcupine_status_text}")
        ax.set_ylabel("Amplitude (int16)")
        ax.set_xlabel("Sample")
        
        # Status text on plot
        status_text_obj = ax.text(0.05, 0.95, "", transform=ax.transAxes, verticalalignment='top')

        # Update function for animation
        def update(frame):
            nonlocal porcupine_buffer
            try:
                # Read audio data
                indata, overflow = stream.read(self.MIC_BLOCKSIZE)
                if indata is not None and indata.size > 0:
                    # Flatten (samples, 1) -> (samples,)
                    data = indata.flatten()
                    line.set_ydata(data)
                    
                    # Update status text with RMS (volume)
                    rms = np.sqrt(np.mean(data.astype(np.float32)**2))
                    db = 20 * np.log10(rms + 1e-9)
                    
                    wakeword_detected = ""
                    
                    # Run Porcupine logic matching server
                    if porcupine is not None:
                         # Convert to bytes
                         raw_bytes = data.tobytes()
                         porcupine_buffer.extend(raw_bytes)
                         
                         while len(porcupine_buffer) >= porcupine_frame_bytes:
                             chunk = bytes(porcupine_buffer[:porcupine_frame_bytes])
                             del porcupine_buffer[:porcupine_frame_bytes]
                             
                             pcm = np.frombuffer(chunk, dtype=np.int16).tolist()
                             keyword_index = porcupine.process(pcm)
                             if keyword_index >= 0:
                                 print(f"[Test] 🌟 WAKE WORD DETECTED! (Index {keyword_index})")
                                 wakeword_detected = "🌟 JARVIS DETECTED!"
                                 # Flash the plot title or something visual
                                 ax.set_title(f"Microphone Input: {device_name}\n🌟 WAKE WORD DETECTED! 🌟")
                    
                    status_text_obj.set_text(f"RMS: {db:.1f} dB\n{wakeword_detected}")

                return line, status_text_obj
            except Exception as e:
                print(f"Stream read error: {e}")
                return line, status_text_obj

        ani = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
        plt.show()

    def _run_text_vu_meter(self):
        """Fallback: Text-based VU meter in terminal."""
        print("[Test] Running text-based VU meter (Ctrl+C to stop)...")
        stream, device_name = self.get_working_stream()
        
        try:
            while True:
                indata, overflow = stream.read(self.MIC_BLOCKSIZE)
                if indata is not None:
                    # Calculate RMS amplitude
                    data = indata.flatten().astype(np.float32)
                    rms = np.sqrt(np.mean(data**2))
                    
                    # Logarithmic scale for VU meter
                    db = 20 * np.log10(rms + 1e-9)
                    # Normalize roughly -60dB to 0dB -> 0 to 50 bars
                    bars = int((db + 60) / 60 * 50)
                    bars = max(0, min(50, bars))
                    
                    bar_str = "█" * bars
                    sys.stdout.write(f"\r[{bar_str:<50}] {db:.1f} dB")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n[Test] Stopped.")

if __name__ == '__main__':
    # If run directly, run the plot test specifically
    suite = unittest.TestSuite()
    suite.addTest(TestMicrophoneInput('test_microphone_plot'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
