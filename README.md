

***

# AI Video Masker Pro (SAM 2)
A professional-grade rotoscoping toolkit for **DaVinci Resolve (Free/Studio)**. This tool uses Meta's **Segment Anything Model 2 (SAM 2)** to generate high-quality masks, optimized for **Apple Silicon (M1/M2/M3)** via Metal (MPS) acceleration.

## 🚀 Choose Your Workflow
This repository provides two distinct workflows depending on your needs:

### 1. Video-to-Video Workflow (`video_masker_v1.py`)
**Best for:** Speed and simplicity. 
- **Input:** A single video file (`.mp4` or `.mov`).
- **Output:** A single masked video file (`.mp4`).
- **Key Features:** 
    - Automated FPS and Resolution detection.
    - **Feathering Slider:** Built-in mask softness adjustment.
    - **UI Status Bar:** On-screen instructions and processing feedback.
    - **Lag-Free Interaction:** Optimized code that responds instantly to clicks.

### 2. Image Sequence Workflow (`selector_v3.py`)
**Best for:** High-precision VFX work or handling complex frame numbers.
- **Input:** A folder of PNG, TIFF, or JPEG images.
- **Output:** A folder of B&W JPEG masks.
- **Key Features:**
    - **Pre-Processor:** Automatically strips timeline prefixes (e.g., `Timeline 1_`) and zero-pads frames for perfect sequential sorting.
    - **Auto-Conversion:** Instantly converts PNG/TIFF to JPEG for AI compatibility.
    - **Workspace Janitor:** Automatically clears old masks before starting.

---

## 🛠️ Setup Instructions

### 1. Clone & Environment
```bash
git clone https://github.com/Musa-codelib/Video_masker.git
cd Video_masker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. AI Model Checkpoint
1. Create a `checkpoints` folder.
2. Download `sam2_hiera_small.pt` from the official SAM 2 repository.
3. Place it inside the `checkpoints` folder.

### 3. Folder Structure
Ensure your directory looks like this:
```text
/AI_masker
├── /input_video      <-- Place MP4/MOV here for video workflow
├── /output_video     <-- Resulting MP4 video appears here
├── /raw_frames       <-- Place Image Sequences here for image workflow
├── /masked_output    <-- Resulting JPEG sequence appears here
├── /checkpoints      <-- sam2_hiera_small.pt
```

---

## 📖 Usage Guide

### Using Video-to-Video (`video_masker_v1.py`)
1. Export your clip from DaVinci Resolve into `/input_video`.
2. Run `python video_masker_v1.py`.
3. **Scrub** to the desired frame using the slider.
4. **Left-Click** (Green) to select the object; **Right-Click** (Red) to exclude areas.
5. Adjust the **Feather** slider for soft mask edges.
6. Press **'P'** to propagate and export the final video.

### Using Image Sequence (`selector_v3.py`)
1. Export a PNG/JPEG sequence from DaVinci Resolve into `/raw_frames`.
2. Run `python selector_v3.py`.
3. The script will automatically clean filenames and convert PNGs to JPEGs.
4. Select your object and press **'P'** to process.
5. The masks will be saved as a sequence in `/masked_output`.

---

## 🎨 DaVinci Resolve Integration (Fusion)
Regardless of the workflow, follow these steps in the **Fusion Page**:
1. Connect the output of the Mask (MediaIn2) to the **Blue (Effect Mask)** input of your footage (MediaIn1).
2. In the **MediaIn1 Inspector -> Settings**:
    - Change **Channel** to **Luminance**.
    - Change **Mapping Mode** to **Stretch**.
3. If using the Image Sequence, ensure the "Start Timecode" in **Clip Attributes** is set to `00:00:00:00`.

---

## ⌨️ Controls Summary
| Key | Action |
| :--- | :--- |
| **Left-Click** | Add selection point (Green) |
| **Right-Click** | Add exclusion point (Red) |
| **R** | Reset all selections |
| **P** | Start AI Processing & Export |
| **Q** | Quit Application |

***

### How to update your GitHub with this:
1. Copy the text above and save it as `README.md` in your folder.
2. Run these commands:
```bash
git add README.md selector_v3.py video_masker_v1.py
git commit -m "Updated README and added both Image and Video workflows"
git push origin main
```