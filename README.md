# Mk Masker

AI-powered rotoscoping tool using Meta's SAM 2/2.1. Creates video masks and alpha cutouts on Apple Silicon.

## Quick Start

```bash
# Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Download model weights to checkpoints/
# - sam2.1_hiera_tiny.pt (Lite) or sam2_hiera_small.pt (Pro)

# Run
python main.py
```

## Project Structure

```
├── main.py                 # GUI entry point
├── engine.py               # Core Radiating Bundle Engine
├── video_masker_v7.py      # Standalone script (SAM 2.1 Tiny)
├── configs/                # Model YAML configs
├── assets/                 # Logo files
├── build/                  # PyInstaller specs
├── checkpoints/            # Model weights (.gitignored)
└── archive/                # Previous versions
```

## Usage

1. Launch `python main.py`
2. Select input video (.mp4/.mov)
3. Choose output folder
4. Pick mode: **ProRes 4444** (alpha) or **B&W Mask**
5. Click **LAUNCH MK SELECTOR**
6. In selector window:
   - **Left-click**: Add selection (green)
   - **Right-click**: Add exclusion (red)
   - **R**: Reset | **P**: Process | **Q**: Quit

## Builds

```bash
# Lite (SAM 2.1 Tiny - faster)
pyinstaller "build/Mk Masker Lite.spec"

# Pro (SAM 2 Small - more accurate)
pyinstaller "build/Mk Masker Pro 1.2.spec"
```

## Requirements

- macOS 13.0+ on Apple Silicon
- Python 3.13
- FFmpeg (for development; bundled in app)
