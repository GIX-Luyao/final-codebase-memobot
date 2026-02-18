"""
Save and reuse a denoising setup (FFT, noise spectrum, gate params) for real-time
audio streams. The setup is built once from a noise sample (e.g. from a reference
video), saved to disk, then loaded and applied to streaming chunks without
re-estimating noise.

Matches the spectral gating used in tests/test_audio_denoise.py (noisereduce
stationary mode) so the same profile can be built there and used here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from scipy.signal import istft, resample, stft

# Defaults aligned with test_audio_denoise.py (fan removal: strong gating, long profile, fine FFT)
# Real-time: use first 0.7 s as fan noise, then strip it from the rest of the stream.
DEFAULT_N_FFT = 8192
DEFAULT_PROP_DECREASE = 0.97
DEFAULT_N_STD_THRESH = 1.5
DEFAULT_NOISE_DURATION_SEC = 0.7
DEFAULT_FREQ_MASK_SMOOTH_HZ = 500
DEFAULT_TIME_MASK_SMOOTH_MS = 50
TOP_DB = 80.0


def _amp_to_db(x: np.ndarray, top_db: float = TOP_DB) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    x_db = 20 * np.log10(np.abs(x) + eps)
    return np.maximum(x_db, np.nanmax(x_db) - top_db)


def _smoothing_filter(n_grad_freq: int, n_grad_time: int) -> np.ndarray:
    f = np.outer(
        np.concatenate([
            np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
            np.linspace(1, 0, n_grad_freq + 2),
        ])[1:-1],
        np.concatenate([
            np.linspace(0, 1, n_grad_time + 1, endpoint=False),
            np.linspace(1, 0, n_grad_time + 2),
        ])[1:-1],
    )
    return f / np.sum(f)


@dataclass
class DenoisingProfile:
    """
    Immutable denoising setup: sample rate, FFT params, noise statistics, and gate params.
    Built once from a noise sample; applied to any number of chunks.
    """
    sr: int
    n_fft: int
    win_length: int
    hop_length: int
    mean_freq_noise: np.ndarray  # shape (n_fft//2 + 1,)
    std_freq_noise: np.ndarray
    n_std_thresh_stationary: float
    prop_decrease: float
    freq_mask_smooth_hz: int | None
    time_mask_smooth_ms: int | None
    # Derived (not saved): noise_thresh, optional smoothing filter
    noise_thresh: np.ndarray = field(repr=False)
    smooth_mask: bool = False
    _smoothing_filter: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_freq(self) -> int:
        return self.mean_freq_noise.shape[0]


def build_profile_from_noise(
    noise_audio: np.ndarray,
    sr: int,
    *,
    n_fft: int = DEFAULT_N_FFT,
    win_length: int | None = None,
    hop_length: int | None = None,
    n_std_thresh_stationary: float = DEFAULT_N_STD_THRESH,
    prop_decrease: float = DEFAULT_PROP_DECREASE,
    freq_mask_smooth_hz: int | None = DEFAULT_FREQ_MASK_SMOOTH_HZ,
    time_mask_smooth_ms: int | None = DEFAULT_TIME_MASK_SMOOTH_MS,
) -> DenoisingProfile:
    """
    Build a denoising profile from a noise-only segment (e.g. first N seconds of
    a reference recording). Use this profile with apply_profile() for real-time
    denoising.

    Args:
        noise_audio: Mono (samples,) or stereo (samples, channels), float or int16.
        sr: Sample rate in Hz.
        n_fft: FFT size (power of two recommended).
        win_length: STFT window length; defaults to n_fft.
        hop_length: STFT hop; defaults to win_length // 4.
        n_std_thresh_stationary: Threshold = mean + n_std * std per frequency bin.
        prop_decrease: Gate strength (0–1).
        freq_mask_smooth_hz: Frequency smoothing of the mask (Hz); None = no smooth.
        time_mask_smooth_ms: Time smoothing of the mask (ms); None = no smooth.

    Returns:
        DenoisingProfile ready to save or use with apply_profile().
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    # Ensure float, mono for noise stats (collapse channels like noisereduce)
    if noise_audio.ndim == 1:
        y = np.asarray(noise_audio, dtype=np.float64)
        if noise_audio.dtype == np.int16:
            y = y / 32768.0
    else:
        y = np.asarray(noise_audio, dtype=np.float64)
        if noise_audio.dtype == np.int16:
            y = y / 32768.0
        y = np.mean(y, axis=1)  # mono

    # STFT of noise
    _, _, Zxx = stft(
        y,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        padded=False,
    )
    noise_stft_db = _amp_to_db(Zxx)
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh_stationary

    # Optional smoothing filter
    smooth_mask = False
    smoothing_filter = None
    if (freq_mask_smooth_hz is not None or time_mask_smooth_ms is not None):
        n_grad_freq = 1 if freq_mask_smooth_hz is None else max(
            1, int(freq_mask_smooth_hz / (sr / (n_fft / 2)))
        )
        n_grad_time = 1 if time_mask_smooth_ms is None else max(
            1, int(time_mask_smooth_ms / ((hop_length / sr) * 1000))
        )
        if n_grad_freq > 1 or n_grad_time > 1:
            smooth_mask = True
            smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)

    return DenoisingProfile(
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        mean_freq_noise=mean_freq_noise,
        std_freq_noise=std_freq_noise,
        n_std_thresh_stationary=n_std_thresh_stationary,
        prop_decrease=prop_decrease,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms,
        noise_thresh=noise_thresh,
        smooth_mask=smooth_mask,
        _smoothing_filter=smoothing_filter,
    )


def _apply_gate_to_channel(
    profile: DenoisingProfile,
    channel: np.ndarray,
) -> np.ndarray:
    """Apply spectral gate to one channel; return denoised 1D array."""
    _, _, Zxx = stft(
        channel,
        nperseg=profile.win_length,
        noverlap=profile.win_length - profile.hop_length,
        nfft=profile.n_fft,
        padded=False,
    )
    sig_stft_db = _amp_to_db(Zxx)
    db_thresh = np.broadcast_to(
        profile.noise_thresh.reshape(-1, 1),
        sig_stft_db.shape,
    )
    sig_mask = (sig_stft_db > db_thresh).astype(np.float64)
    sig_mask = sig_mask * profile.prop_decrease + (1.0 - profile.prop_decrease)

    if profile.smooth_mask and profile._smoothing_filter is not None:
        from scipy.signal import fftconvolve
        sig_mask = fftconvolve(sig_mask, profile._smoothing_filter, mode="same")

    Zxx_denoised = Zxx * sig_mask
    _, denoised = istft(
        Zxx_denoised,
        nperseg=profile.win_length,
        noverlap=profile.win_length - profile.hop_length,
        nfft=profile.n_fft,
    )
    # Preserve exact length for streaming: trim or zero-pad to match input
    target_len = len(channel)
    if len(denoised) >= target_len:
        out = denoised[:target_len].astype(np.float64)
    else:
        out = np.zeros(target_len, dtype=np.float64)
        out[:len(denoised)] = denoised
    return out


def apply_profile(
    profile: DenoisingProfile,
    audio: np.ndarray,
    *,
    return_float: bool = False,
) -> np.ndarray:
    """
    Denoise audio using a saved profile. Safe for real-time streaming: pass
    chunks of any length (longer than ~hop_length is better). No file I/O.

    Args:
        profile: From build_profile_from_noise() or load_profile().
        audio: Mono (samples,) or stereo (samples, channels), float or int16.
        return_float: If True, return float in [-1, 1]; else same dtype as input.

    Returns:
        Denoised array, same shape as audio (possibly one sample less per channel
        due to STFT/ISTFT). Dtype per return_float.
    """
    if audio.ndim == 1:
        chans = [np.asarray(audio, dtype=np.float64)]
        if audio.dtype == np.int16:
            chans[0] = chans[0] / 32768.0
        single = True
    else:
        chans = [
            np.asarray(audio[:, c], dtype=np.float64)
            for c in range(audio.shape[1])
        ]
        if audio.dtype == np.int16:
            chans = [c / 32768.0 for c in chans]
        single = False

    out_chans = [_apply_gate_to_channel(profile, c) for c in chans]

    if not return_float:
        out_chans = [(np.clip(c, -1.0, 1.0) * 32767).astype(np.int16) for c in out_chans]

    if single:
        return out_chans[0]
    # All channels already length-matched to input in _apply_gate_to_channel
    return np.stack(out_chans, axis=1)


def apply_profile_to_bytes(
    profile: DenoisingProfile,
    audio_bytes: bytes,
    channels: int = 1,
    sample_width: int = 2,
    **kwargs: bool,
) -> bytes:
    """Denoise raw PCM bytes using the profile. Returns denoised PCM bytes."""
    if sample_width != 2:
        raise ValueError("Only sample_width=2 (int16) is supported")
    frame = channels * sample_width
    if len(audio_bytes) % frame != 0:
        audio_bytes = audio_bytes[: (len(audio_bytes) // frame) * frame]
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    if channels > 1:
        arr = arr.reshape(-1, channels)
    out = apply_profile(profile, arr, return_float=False, **kwargs)
    return out.astype(np.int16).tobytes()


def build_profile_from_video(
    video_path: str | Path,
    noise_duration_sec: float = DEFAULT_NOISE_DURATION_SEC,
    target_sr: int | None = None,
    **kwargs: float | int | None,
) -> DenoisingProfile:
    """
    Build a denoising profile from the first N seconds of a reference video.
    Requires moviepy. Use save_profile() to persist for real-time use.

    Args:
        video_path: Path to reference video (e.g. full clip with fan-only lead-in).
        noise_duration_sec: Seconds of leading audio to use as noise.
        target_sr: If set (e.g. 16000 for robot), resample noise to this rate before
            building the profile. Use for real-time 16 kHz streams.
        **kwargs: Passed to build_profile_from_noise (n_fft, prop_decrease, etc.).

    Returns:
        DenoisingProfile ready to save or apply.
    """
    try:
        from moviepy import VideoFileClip
    except ImportError as e:
        raise ImportError("moviepy is required for build_profile_from_video") from e

    path = Path(video_path)
    clip = VideoFileClip(str(path))
    fps = getattr(clip.audio, "fps", 44100)
    audio = clip.audio.to_soundarray(fps=fps)
    clip.close()
    if audio.ndim == 1:
        noise = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    else:
        noise = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    sr = int(round(fps))
    n = min(int(sr * noise_duration_sec), noise.shape[0])
    if noise.ndim == 1:
        noise_segment = noise[:n]
    else:
        noise_segment = noise[:n, :]

    if target_sr is not None and target_sr != sr:
        # Resample to target_sr for robot real-time (e.g. 16 kHz)
        num_samples = int(len(noise_segment) * target_sr / sr)
        if noise_segment.ndim == 1:
            noise_segment = resample(noise_segment.astype(np.float64), num_samples).astype(np.int16)
        else:
            noise_segment = np.column_stack([
                resample(noise_segment[:, c].astype(np.float64), num_samples) for c in range(noise_segment.shape[1])
            ]).astype(np.int16)
        sr = target_sr

    return build_profile_from_noise(noise_segment, sr, **kwargs)


def denoise_video_file(
    video_path: str | Path,
    output_path: str | Path,
    noise_duration_sec: float = DEFAULT_NOISE_DURATION_SEC,
    **kwargs: float | int | None,
) -> Path:
    """
    Build a fan-noise profile from the first N seconds of the video, strip that
    noise from the full track, and save a new video with denoised audio.

    Args:
        video_path: Input video (first noise_duration_sec used as noise sample).
        output_path: Path for the output video (e.g. .../clip_denoised.mp4).
        noise_duration_sec: Seconds of leading audio to use as noise (default 0.7).
        **kwargs: Passed to build_profile_from_noise (n_fft, prop_decrease, etc.).

    Returns:
        Path to the written output file.
    """
    try:
        from moviepy import VideoFileClip
        from moviepy.audio.AudioClip import AudioArrayClip
    except ImportError as e:
        raise ImportError("moviepy is required for denoise_video_file") from e

    video_path = Path(video_path)
    output_path = Path(output_path)

    # Load full audio and video
    clip = VideoFileClip(str(video_path))
    fps = getattr(clip.audio, "fps", 44100)
    sr = int(round(fps))
    audio = clip.audio.to_soundarray(fps=fps)
    clip.close()

    if audio.ndim == 1:
        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    else:
        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    # Noise segment: first N seconds
    n_noise = min(int(sr * noise_duration_sec), audio_int16.shape[0])
    if audio_int16.ndim == 1:
        noise_segment = audio_int16[:n_noise]
    else:
        noise_segment = audio_int16[:n_noise, :]

    profile = build_profile_from_noise(noise_segment, sr, **kwargs)
    denoised = apply_profile(profile, audio_int16, return_float=True)

    # AudioArrayClip expects (samples, channels) float in [-1, 1]
    denoised = np.clip(denoised, -1.0, 1.0).astype(np.float64)
    if denoised.ndim == 1:
        denoised = denoised[:, np.newaxis]

    clean_audio_clip = AudioArrayClip(denoised, fps=sr)
    video_clip = VideoFileClip(str(video_path))
    final = video_clip.with_audio(clean_audio_clip)
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    clean_audio_clip.close()
    video_clip.close()
    final.close()
    return output_path


# ---- Save / load profile (exact FFT + frequency setup) ----

PROFILE_JSON = "denoising_profile.json"
PROFILE_NPZ = "denoising_profile.npz"


def save_profile(profile: DenoisingProfile, path: str | Path) -> Path:
    """
    Save the exact denoising setup to a directory: JSON for scalar params and
    .npz for arrays. Use load_profile() to restore.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    meta = {
        "sr": profile.sr,
        "n_fft": profile.n_fft,
        "win_length": profile.win_length,
        "hop_length": profile.hop_length,
        "n_std_thresh_stationary": profile.n_std_thresh_stationary,
        "prop_decrease": profile.prop_decrease,
        "freq_mask_smooth_hz": profile.freq_mask_smooth_hz,
        "time_mask_smooth_ms": profile.time_mask_smooth_ms,
    }
    with open(path / PROFILE_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    np.savez(
        path / PROFILE_NPZ,
        mean_freq_noise=profile.mean_freq_noise,
        std_freq_noise=profile.std_freq_noise,
        noise_thresh=profile.noise_thresh,
    )
    return path


def get_default_profile_dir() -> Path:
    """Default directory for saved profile (ingest_pipeline data)."""
    return Path(__file__).resolve().parents[1] / "ingest_pipeline" / "data" / "denoising_profile"


def get_robot_profile_dir() -> Path:
    """Directory for 16 kHz profile used by robot real-time audio (mac_master_v10)."""
    return Path(__file__).resolve().parents[1] / "ingest_pipeline" / "data" / "denoising_profile_16k"


def load_profile(path: str | Path) -> DenoisingProfile:
    """Load a denoising profile from a directory saved with save_profile()."""
    path = Path(path)
    with open(path / PROFILE_JSON) as f:
        meta = json.load(f)
    npz = np.load(path / PROFILE_NPZ)
    mean_freq_noise = npz["mean_freq_noise"]
    std_freq_noise = npz["std_freq_noise"]
    noise_thresh = npz["noise_thresh"]

    win_length = int(meta["win_length"])
    hop_length = int(meta["hop_length"])
    sr = int(meta["sr"])
    n_fft = int(meta["n_fft"])
    freq_mask_smooth_hz = meta.get("freq_mask_smooth_hz")
    time_mask_smooth_ms = meta.get("time_mask_smooth_ms")
    if freq_mask_smooth_hz is not None:
        freq_mask_smooth_hz = int(freq_mask_smooth_hz)
    if time_mask_smooth_ms is not None:
        time_mask_smooth_ms = int(time_mask_smooth_ms)

    smooth_mask = False
    smoothing_filter = None
    if freq_mask_smooth_hz or time_mask_smooth_ms:
        n_grad_freq = 1 if not freq_mask_smooth_hz else max(
            1, int(freq_mask_smooth_hz / (sr / (n_fft / 2)))
        )
        n_grad_time = 1 if not time_mask_smooth_ms else max(
            1, int(time_mask_smooth_ms / ((hop_length / sr) * 1000))
        )
        if n_grad_freq > 1 or n_grad_time > 1:
            smooth_mask = True
            smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)

    return DenoisingProfile(
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        mean_freq_noise=mean_freq_noise,
        std_freq_noise=std_freq_noise,
        n_std_thresh_stationary=float(meta["n_std_thresh_stationary"]),
        prop_decrease=float(meta["prop_decrease"]),
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms,
        noise_thresh=noise_thresh,
        smooth_mask=smooth_mask,
        _smoothing_filter=smoothing_filter,
    )


if __name__ == "__main__":
    import argparse
    data_dir = Path(__file__).resolve().parents[1] / "ingest_pipeline" / "data"
    default_video = data_dir / "clip_20260217_201526.mp4"
    parser = argparse.ArgumentParser(
        description="Build denoising profile and/or output denoised video (noise = first 0.7 s)."
    )
    parser.add_argument("video", nargs="?", default=None, help="Input video (first N sec = fan noise)")
    parser.add_argument("-o", "--out-dir", default=None, help="Output directory for profile (build-profile mode)")
    parser.add_argument(
        "--denoise-out",
        default=None,
        metavar="PATH",
        help="Output path for denoised video; default: input_denoised.mp4 in same dir when using default video",
    )
    parser.add_argument("--profile-only", action="store_true", help="Only build/save profile, do not write denoised video")
    parser.add_argument("--robot", action="store_true", help="Build 16 kHz profile for robot (saves to denoising_profile_16k)")
    parser.add_argument("--noise-sec", type=float, default=DEFAULT_NOISE_DURATION_SEC, help="Seconds of lead-in to use as noise (default 0.7)")
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--prop-decrease", type=float, default=DEFAULT_PROP_DECREASE)
    args = parser.parse_args()
    video_path = args.video or str(default_video)
    if not Path(video_path).exists():
        raise SystemExit(f"Video not found: {video_path}")

    denoise_out = args.denoise_out
    if denoise_out is None and not args.profile_only:
        # Default: save denoised version next to input
        video_path_obj = Path(video_path)
        denoise_out = str(video_path_obj.parent / f"{video_path_obj.stem}_denoised{video_path_obj.suffix}")

    if denoise_out is not None:
        # Denoise video: first 0.7 s = noise, strip from full track, save new file
        out_path = denoise_video_file(
            video_path,
            denoise_out,
            noise_duration_sec=args.noise_sec,
            n_fft=args.n_fft,
            prop_decrease=args.prop_decrease,
        )
        print(f"Denoised video saved to {out_path}")
    else:
        # Build and save profile only
        if args.robot:
            out_dir = args.out_dir or str(get_robot_profile_dir())
            target_sr = 16000
        else:
            out_dir = args.out_dir or str(get_default_profile_dir())
            target_sr = None
        profile = build_profile_from_video(
            video_path,
            noise_duration_sec=args.noise_sec,
            target_sr=target_sr,
            n_fft=args.n_fft,
            prop_decrease=args.prop_decrease,
        )
        save_profile(profile, out_dir)
        print(f"Profile saved to {out_dir} (sr={profile.sr})")
