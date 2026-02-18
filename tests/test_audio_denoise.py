"""
Standalone test: denoise incoming client audio using spectral gating.

Denoising is byte/array-based (no file I/O in the core):
- denoise_audio(audio_array, sr, ...)  — in-memory numpy in/out
- denoise_audio_bytes(audio_bytes, sr, ...) — raw PCM bytes in/out for streams

A file-based video helper is also provided for the script use case.

Run with:
  pytest tests/test_audio_denoise.py -v
  python tests/test_audio_denoise.py   # runs as script and writes output file

Dependencies: moviepy, noisereduce, scipy, numpy (see pyproject.toml).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Optional imports for denoising; test is skipped if any are missing
VideoFileClip = AudioFileClip = AudioArrayClip = None
wavfile = nr = None
_import_error: BaseException | None = None
try:
    from moviepy import AudioFileClip, VideoFileClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from scipy.io import wavfile
    import noisereduce as nr
except ImportError as e:
    VideoFileClip = AudioFileClip = AudioArrayClip = wavfile = nr = None
    _import_error = e


# Reference clip: used to build the noise profile (full video, more silence/fan).
REFERENCE_CLIP_PATH = Path(
    "/Users/jasonyang/Documents/Development/memobot/memobot/ingest_pipeline/data/clip_20260217_170454.mp4"
)
# Original clip: denoised using the reference profile, output written for this.
CLIP_PATH = Path(
    "/Users/jasonyang/Documents/Development/memobot/memobot/ingest_pipeline/data/clip_20260217_170353.mp4"
)
# Output written next to original, with this suffix
OUTPUT_SUFFIX = "_denoised_enhanced.mp4"


def denoise_audio(
    audio: np.ndarray,
    sr: int,
    *,
    noise_part: np.ndarray | None = None,
    noise_duration_sec: float = 0.7,
    prop_decrease: float = 0.97,
    normalize_peak: float = 0.9,
    return_float: bool = False,
    n_fft: int = 4096,
) -> np.ndarray:
    """
    Denoise audio in memory via spectral gating (no file I/O).

    Uses the first `noise_duration_sec` seconds as the noise profile if
    `noise_part` is not provided, then reduces that spectrum and normalizes.
    Tuned for constant hum (e.g. fan): longer noise profile, stronger reduction,
    larger FFT for better frequency resolution.

    Args:
        audio: Mono (samples,) or stereo (samples, channels), int16 or float.
        sr: Sample rate in Hz.
        noise_part: Optional pre-extracted noise segment; same layout as audio
            (channels last). If None, uses the first noise_duration_sec of audio.
        noise_duration_sec: Length of leading segment to use as noise profile.
        prop_decrease: Noise reduction strength (0–1). 0.72 targets constant hum.
        normalize_peak: Peak level after normalization (e.g. 0.9).
        return_float: If True, return float in [-1, 1]; else int16.
        n_fft: FFT size; larger values give finer frequency resolution (helps hum).

    Returns:
        Denoised array, same shape as input, int16 or float per return_float.
    """
    # Layout for noisereduce: (channels, samples)
    if audio.ndim == 1:
        data = audio.astype(np.float64) if audio.dtype != np.float64 else audio.copy()
        if data.dtype == np.int16:
            data = data / 32768.0
        data = data[np.newaxis, :]
        single_channel = True
    else:
        data = audio.astype(np.float64) if audio.dtype != np.float64 else audio.copy()
        if audio.dtype == np.int16:
            data = data / 32768.0
        data = data.T  # (samples, channels) -> (channels, samples)
        single_channel = False

    if noise_part is not None:
        if noise_part.ndim == 1:
            npart = noise_part.astype(np.float64)
            if noise_part.dtype == np.int16:
                npart = npart / 32768.0
            npart = npart[np.newaxis, :]
        else:
            npart = noise_part.T.astype(np.float64)
            if noise_part.dtype == np.int16:
                npart = npart / 32768.0
    else:
        n_noise = min(int(sr * noise_duration_sec), data.shape[1])
        npart = data[:, :n_noise]

    reduced = nr.reduce_noise(
        y=data,
        sr=sr,
        y_noise=npart,
        prop_decrease=prop_decrease,
        stationary=True,
        n_fft=n_fft,
    )

    max_val = np.max(np.abs(reduced))
    if max_val > 0:
        enhanced = reduced / max_val * normalize_peak
    else:
        enhanced = reduced

    if not return_float:
        enhanced = (enhanced * 32767).astype(np.int16)

    if single_channel:
        out = enhanced[0]
    else:
        out = enhanced.T  # (channels, samples) -> (samples, channels)
    return out


def denoise_audio_bytes(
    audio_bytes: bytes,
    sample_rate: int,
    *,
    channels: int = 1,
    sample_width: int = 2,
    **kwargs: object,
) -> bytes:
    """
    Denoise raw PCM audio bytes (no file I/O). Suitable for streams.

    Args:
        audio_bytes: Raw PCM, little-endian int16 (sample_width=2).
        sample_rate: Sample rate in Hz.
        channels: Number of channels.
        sample_width: Bytes per sample (only 2 supported for int16).
        **kwargs: Passed to denoise_audio (noise_duration_sec, prop_decrease, etc.).

    Returns:
        Denoised PCM bytes, same format and length as input (within frame alignment).
    """
    if sample_width != 2:
        raise ValueError("Only sample_width=2 (int16) is supported")
    dtype = np.int16
    frame_size = channels * sample_width
    if len(audio_bytes) % frame_size != 0:
        audio_bytes = audio_bytes[: (len(audio_bytes) // frame_size) * frame_size]
    samples = np.frombuffer(audio_bytes, dtype=dtype)
    if channels > 1:
        audio = samples.reshape(-1, channels)
    else:
        audio = samples
    out = denoise_audio(audio, sample_rate, return_float=False, **kwargs)
    return out.astype(dtype).tobytes()


def _audio_array_from_video(video_path: Path):
    """Load video, return (audio_int16, sr). Audio shape (samples,) or (samples, channels)."""
    video_clip = VideoFileClip(str(video_path))
    audio_clip = video_clip.audio
    fps = getattr(audio_clip, "fps", 44100)
    audio_array = audio_clip.to_soundarray(fps=fps)
    if audio_array.ndim == 1:
        audio_int16 = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
    else:
        audio_int16 = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
    video_clip.close()
    return audio_int16, int(round(fps))


def extract_noise_profile_from_video(
    video_path: str | Path,
    noise_duration_sec: float = 0.7,
) -> tuple[np.ndarray, int]:
    """
    Build denoising setup from a reference video: extract noise profile (first N sec).

    Returns (noise_part, sr). noise_part has same layout as denoise_audio expects
    for noise_part: (samples,) mono or (samples, channels) stereo.
    """
    video_path = Path(video_path)
    audio, sr = _audio_array_from_video(video_path)
    n = min(int(sr * noise_duration_sec), audio.shape[0])
    if audio.ndim == 1:
        noise_part = audio[:n].copy()
    else:
        noise_part = audio[:n, :].copy()
    return noise_part, sr


def clean_and_enhance_video_with_reference(
    reference_video_path: str | Path,
    source_video_path: str | Path,
    output_path: str | Path,
    *,
    noise_duration_sec: float = 3,
    prop_decrease: float = 0.97,
    normalize_peak: float = 0.9,
) -> Path:
    """
    Get full denoising setup from reference video, then denoise source video.

    Uses reference_video_path to extract the noise profile (first noise_duration_sec),
    then denoises the audio of source_video_path with that profile and muxes
    cleaned audio back onto the source video, writing to output_path.
    """
    reference_video_path = Path(reference_video_path)
    source_video_path = Path(source_video_path)
    output_path = Path(output_path)

    # 1. Build denoising setup from reference (full video)
    noise_part, ref_sr = extract_noise_profile_from_video(
        reference_video_path, noise_duration_sec=noise_duration_sec
    )

    # 2. Load source (original) video and get its audio
    source_audio, source_sr = _audio_array_from_video(source_video_path)
    if source_sr != ref_sr:
        # Resample noise profile to source sr if needed (simple repeat/decimate for test)
        raise ValueError(
            f"Reference sr {ref_sr} != source sr {source_sr}; resampling not implemented"
        )

    # 3. Denoise source audio using reference noise profile
    denoised = denoise_audio(
        source_audio,
        source_sr,
        noise_part=noise_part,
        prop_decrease=prop_decrease,
        normalize_peak=normalize_peak,
        return_float=False,
        n_fft=8192,
    )

    # 4. Mux cleaned audio back onto source video and write
    clean_float = denoised.astype(np.float64) / 32768.0
    if clean_float.ndim == 1:
        clean_float = clean_float[:, np.newaxis]
    clean_audio_clip = AudioArrayClip(clean_float, fps=source_sr)

    video_clip = VideoFileClip(str(source_video_path))
    final_video = video_clip.with_audio(clean_audio_clip)
    final_video.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    clean_audio_clip.close()
    video_clip.close()

    return output_path


def clean_and_enhance_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    noise_duration_sec: float = 0.7,
    prop_decrease: float = 0.97,
    normalize_peak: float = 0.9,
) -> Path:
    """
    Denoise video audio and mux back (uses in-memory denoise; only video in/out touch disk).

    Reads video from file, extracts audio to an array in memory, runs denoise_audio,
    then muxes cleaned audio back and writes the output file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    video_clip = VideoFileClip(str(input_path))
    audio_clip = video_clip.audio
    # Get audio as array in memory (no temp WAV for denoising)
    fps = getattr(audio_clip, "fps", 44100)
    audio_array = audio_clip.to_soundarray(fps=fps)
    # (samples, channels) float [-1, 1]; convert to int16 for denoise_audio
    if audio_array.ndim == 1:
        audio_int16 = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
    else:
        audio_int16 = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
    sr = int(round(fps))

    denoised = denoise_audio(
        audio_int16,
        sr,
        noise_duration_sec=noise_duration_sec,
        prop_decrease=prop_decrease,
        normalize_peak=normalize_peak,
        return_float=False,
        n_fft=8192,
    )
    # AudioArrayClip expects (samples, channels) float in [-1, 1]
    clean_float = denoised.astype(np.float64) / 32768.0
    if clean_float.ndim == 1:
        clean_float = clean_float[:, np.newaxis]

    clean_audio_clip = AudioArrayClip(clean_float, fps=sr)
    final_video = video_clip.with_audio(clean_audio_clip)
    final_video.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    clean_audio_clip.close()
    video_clip.close()

    return output_path


@pytest.mark.skipif(
    nr is None,
    reason="noisereduce not installed",
)
def test_denoise_audio_bytes_in_memory():
    """Denoising is in-memory: array and bytes APIs produce output without files."""
    sr = 16000
    duration_sec = 1.0
    n = int(sr * duration_sec)
    # Fake noise (constant) + tone (signal)
    rng = np.random.default_rng(42)
    noise = (rng.uniform(-0.1, 0.1, n) * 32767).astype(np.int16)
    tone = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, n, endpoint=False)) * 0.3 * 32767).astype(np.int16)
    audio = (noise + tone).astype(np.int16)
    # Array API
    out_array = denoise_audio(audio, sr, noise_duration_sec=0.2, return_float=False)
    assert out_array.shape == audio.shape
    assert out_array.dtype == np.int16
    # Bytes API
    audio_bytes = audio.tobytes()
    out_bytes = denoise_audio_bytes(audio_bytes, sr, channels=1, noise_duration_sec=0.2)
    assert len(out_bytes) == len(audio_bytes)
    out_from_bytes = np.frombuffer(out_bytes, dtype=np.int16)
    assert out_from_bytes.shape == audio.shape


@pytest.mark.skipif(
    VideoFileClip is None or AudioArrayClip is None or nr is None,
    reason="audio denoise deps missing (moviepy, scipy, noisereduce)",
)
def test_denoise_clip_with_reference():
    """Build denoising setup from reference clip (171312), denoise original (171342), assert output."""
    if not REFERENCE_CLIP_PATH.exists():
        pytest.skip(f"Reference clip not found: {REFERENCE_CLIP_PATH}")
    if not CLIP_PATH.exists():
        pytest.skip(f"Clip not found: {CLIP_PATH}")

    with tempfile.TemporaryDirectory(prefix="denoise_out_") as out_dir:
        out_file = Path(out_dir) / "clip_denoised_enhanced.mp4"
        result = clean_and_enhance_video_with_reference(
            REFERENCE_CLIP_PATH,
            CLIP_PATH,
            out_file,
        )
        assert result.exists()
        assert result.stat().st_size > 0


def main() -> None:
    """Build denoising setup from reference (171312), denoise original (171342), write output."""
    if VideoFileClip is None or AudioArrayClip is None or nr is None:
        raise RuntimeError(
            "Install moviepy, noisereduce, and scipy (e.g. uv sync)"
        ) from _import_error

    if not REFERENCE_CLIP_PATH.exists():
        raise FileNotFoundError(f"Reference clip not found: {REFERENCE_CLIP_PATH}")
    if not CLIP_PATH.exists():
        raise FileNotFoundError(f"Clip not found: {CLIP_PATH}")

    output_path = CLIP_PATH.parent / (CLIP_PATH.stem + "_denoised_enhanced.mp4")
    print(f"Building noise profile from: {REFERENCE_CLIP_PATH}")
    print(f"Denoising original video: {CLIP_PATH}")
    clean_and_enhance_video_with_reference(
        REFERENCE_CLIP_PATH,
        CLIP_PATH,
        output_path,
    )
    print(f"Done. Saved to: {output_path}")


if __name__ == "__main__":
    main()
