# Video AI Masker Pro (SAM 2)
A modular AI-powered rotoscoping tool for DaVinci Resolve (Free Version). 

## Features
- Optimized for Apple Silicon (M1/M2/M3) using MPS.
- Video-to-Video workflow (No image sequence bugs).
- Real-time interactive selection (SAM 2).
- Built-in mask feathering and automated pre-processing.

## Setup
1. Clone the repo.
2. Install SAM 2: `pip install git+https://github.com/facebookresearch/segment-anything-2.git`
3. Install requirements: `pip install -r requirements.txt`
4. Place `sam2_hiera_small.pt` in the `checkpoints` folder.

## Usage
1. Export clip from Resolve to `input_video`.
2. Run `python video_masker.py`.
3. Select object, press 'P' to process.
4. Import result from `output_video` back to Resolve.
