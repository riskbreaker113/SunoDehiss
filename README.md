# Suno De-Hiss

Batch removes high-frequency hiss/artifacts from AI-generated music (.wav files) before uploading to DistroKid.

## Setup
```bash
pip install noisereduce scipy numpy soundfile
```

## Usage
1. Put your .wav files in an `input` folder next to the script
2. Run: `python suno_dehiss.py --strength low`
3. Cleaned files appear in `output/`

### Strength options
- `--strength low` — light touch, try first
- `--strength medium` — default, good balance
- `--strength high` — for badly hissy tracks
- `--preview song.wav` — A/B test a single file
