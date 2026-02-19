"""
Suno Audio De-Hiss Batch Processor
===================================
Removes high-frequency hiss/artifacts commonly found in AI-generated music
(Suno, etc.) before uploading to DistroKid or other distributors.

Does NOT require a quiet/silent section to sample noise — uses a general
spectral approach suited to AI-generated audio artifacts.

Usage:
    python suno_dehiss.py                          # Process all .wav in ./input, save to ./output
    python suno_dehiss.py --input my_songs         # Custom input folder
    python suno_dehiss.py --strength medium         # low / medium / high noise reduction
    python suno_dehiss.py --preview song.wav       # Process one file and save A/B comparison

Requirements (install once):
    pip install noisereduce scipy numpy soundfile

Windows note: If 'pip' doesn't work, try 'python -m pip install ...'
"""

import argparse
import os
import sys
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.exit("Missing dependency: soundfile\nRun: pip install soundfile")

try:
    import noisereduce as nr
except ImportError:
    sys.exit("Missing dependency: noisereduce\nRun: pip install noisereduce")

try:
    from scipy.signal import butter, sosfilt
except ImportError:
    sys.exit("Missing dependency: scipy\nRun: pip install scipy")


# ─── Strength Presets ───────────────────────────────────────────────
# Each preset controls:
#   - prop_decrease: how aggressively noise is reduced (0.0 = none, 1.0 = full)
#   - lowpass_freq:  cutoff for gentle low-pass filter (Hz), None = disabled
#   - highshelf_db:  gentle high-shelf attenuation above 14kHz (negative dB)

PRESETS = {
    "low": {
        "prop_decrease": 0.4,
        "lowpass_freq": None,       # no low-pass at low strength
        "highshelf_db": -1.5,
    },
    "medium": {
        "prop_decrease": 0.6,
        "lowpass_freq": 17500,      # gentle rolloff above 17.5kHz
        "highshelf_db": -2.5,
    },
    "high": {
        "prop_decrease": 0.8,
        "lowpass_freq": 16000,      # more aggressive rolloff
        "highshelf_db": -4.0,
    },
}


def apply_lowpass(audio, sr, cutoff_freq, order=5):
    """Apply a Butterworth low-pass filter."""
    nyquist = sr / 2.0
    if cutoff_freq >= nyquist:
        return audio
    normalized_cutoff = cutoff_freq / nyquist
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    if audio.ndim == 1:
        return sosfilt(sos, audio)
    else:
        # Process each channel
        return np.column_stack([sosfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])])


def apply_high_shelf(audio, sr, shelf_freq=14000, gain_db=-2.5):
    """
    Simple high-shelf EQ: attenuate frequencies above shelf_freq.
    Uses a 2nd-order Butterworth high-pass to isolate highs, then scales them down.
    """
    if gain_db >= 0:
        return audio
    nyquist = sr / 2.0
    if shelf_freq >= nyquist:
        return audio

    normalized_freq = shelf_freq / nyquist
    sos = butter(2, normalized_freq, btype='high', output='sos')

    gain_linear = 10 ** (gain_db / 20.0)
    scale = gain_linear - 1.0  # how much to ADD (negative = cut)

    if audio.ndim == 1:
        highs = sosfilt(sos, audio)
        return audio + scale * highs
    else:
        result = audio.copy()
        for ch in range(audio.shape[1]):
            highs = sosfilt(sos, audio[:, ch])
            result[:, ch] = audio[:, ch] + scale * highs
        return result


def process_file(input_path, output_path, preset_name="medium", verbose=True):
    """Process a single .wav file to remove hiss."""
    preset = PRESETS[preset_name]

    if verbose:
        print(f"  Reading: {os.path.basename(input_path)}")

    audio, sr = sf.read(input_path, dtype='float64')

    if verbose:
        channels = "stereo" if audio.ndim > 1 else "mono"
        duration = len(audio) / sr
        print(f"  Format: {sr}Hz, {channels}, {duration:.1f}s")
        print(f"  Applying noise reduction (strength: {preset_name})...")

    # Step 1: Spectral noise reduction (stationary noise — no reference clip needed)
    # noisereduce's stationary mode estimates the noise floor from the full signal
    if audio.ndim == 1:
        cleaned = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=preset["prop_decrease"],
            n_fft=2048,
            freq_mask_smooth_hz=500,
        )
    else:
        # Process stereo channels
        channels = []
        for ch in range(audio.shape[1]):
            ch_cleaned = nr.reduce_noise(
                y=audio[:, ch],
                sr=sr,
                stationary=True,
                prop_decrease=preset["prop_decrease"],
                n_fft=2048,
                freq_mask_smooth_hz=500,
            )
            channels.append(ch_cleaned)
        cleaned = np.column_stack(channels)

    # Step 2: High-shelf EQ — gently tame the high-frequency region
    if verbose:
        print(f"  Applying high-shelf EQ ({preset['highshelf_db']} dB above 14kHz)...")
    cleaned = apply_high_shelf(cleaned, sr, shelf_freq=14000, gain_db=preset["highshelf_db"])

    # Step 3: Low-pass filter (if preset includes one)
    if preset["lowpass_freq"] is not None:
        if verbose:
            print(f"  Applying low-pass filter at {preset['lowpass_freq']}Hz...")
        cleaned = apply_lowpass(cleaned, sr, preset["lowpass_freq"])

    # Prevent clipping
    peak = np.max(np.abs(cleaned))
    if peak > 1.0:
        if verbose:
            print(f"  Normalizing to prevent clipping (peak was {peak:.3f})...")
        cleaned = cleaned / peak * 0.99

    # Write output
    sf.write(output_path, cleaned, sr, subtype='PCM_24')  # 24-bit WAV for distribution
    if verbose:
        print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove hiss/artifacts from Suno AI-generated .wav files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python suno_dehiss.py                         Process ./input -> ./output (medium)
  python suno_dehiss.py --strength low          Lighter touch, preserves more highs
  python suno_dehiss.py --strength high         Aggressive, for badly hissy tracks
  python suno_dehiss.py --input "C:\\Music\\Suno" --output "C:\\Music\\Clean"
  python suno_dehiss.py --preview song.wav      Quick A/B test on one file
        """
    )
    parser.add_argument("--input", default="input",
                        help="Folder containing .wav files (default: ./input)")
    parser.add_argument("--output", default="output",
                        help="Folder for cleaned files (default: ./output)")
    parser.add_argument("--strength", choices=["low", "medium", "high"], default="medium",
                        help="Noise reduction strength (default: medium)")
    parser.add_argument("--preview", metavar="FILE",
                        help="Process a single file for quick A/B comparison")

    args = parser.parse_args()

    print("=" * 60)
    print("  Suno De-Hiss Processor")
    print(f"  Strength: {args.strength}")
    print("=" * 60)

    # Single file preview mode
    if args.preview:
        if not os.path.isfile(args.preview):
            sys.exit(f"File not found: {args.preview}")
        base, ext = os.path.splitext(args.preview)
        out_path = f"{base}_cleaned{ext}"
        print(f"\nPreview mode — processing single file:")
        process_file(args.preview, out_path, args.strength)
        print(f"\nDone! Compare the original and cleaned file side by side.")
        return

    # Batch mode
    if not os.path.isdir(args.input):
        os.makedirs(args.input, exist_ok=True)
        print(f"\nCreated input folder: {os.path.abspath(args.input)}")
        print("Drop your .wav files in there and run this script again.")
        return

    wav_files = [f for f in os.listdir(args.input) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"\nNo .wav files found in: {os.path.abspath(args.input)}")
        print("Drop your .wav files in there and run this script again.")
        return

    os.makedirs(args.output, exist_ok=True)

    print(f"\nFound {len(wav_files)} file(s) in: {os.path.abspath(args.input)}")
    print(f"Output folder: {os.path.abspath(args.output)}\n")

    success = 0
    errors = []

    for i, filename in enumerate(sorted(wav_files), 1):
        print(f"[{i}/{len(wav_files)}] {filename}")
        input_path = os.path.join(args.input, filename)
        output_path = os.path.join(args.output, filename)
        try:
            process_file(input_path, output_path, args.strength)
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((filename, str(e)))
        print()

    print("=" * 60)
    print(f"  Complete: {success}/{len(wav_files)} files processed")
    if errors:
        print(f"  Errors: {len(errors)}")
        for fname, err in errors:
            print(f"    - {fname}: {err}")
    print(f"  Output: {os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
