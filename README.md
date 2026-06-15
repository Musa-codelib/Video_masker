
# AI Video Masker Pro (SAM 2)
A professional-grade rotoscoping toolkit for **DaVinci Resolve (Free/Studio)**. This tool uses Meta's **Segment Anything Model 2 (SAM 2)** to generate high-quality masks, optimized for **Apple Silicon (M1/M2/M3)** via Metal (MPS) acceleration.

## 🚀 Choose Your Workflow

### 1. Image Sequence Workflow
| Script | Use Case | Output |
| :--- | :--- | :--- |
| **`selector_v3.py`** | High-precision VFX / PNG Sequences | B&W JPEG Sequence |

### 2. Video-to-Video Workflow (B&W Masks)
*Best for users who want to use the Fusion "Luma to Alpha" method.*
| Script | UI Style | Output |
| :--- | :--- | :--- |
| **`video_masker_v1.py`** | **Simple:** Minimalist, distraction-free | B&W MP4 Video |
| **`video_masker_v2.py`** | **Advanced:** HUD, instructions & status bar | B&W MP4 Video |

### 3. Pro-Level Cutout Workflow (Transparent Alpha)
*Best for users who want to bypass Fusion. Drag-and-drop transparency directly on the Edit Page.*
| Script | UI Style | Output |
| :--- | :--- | :--- |
| **`video_masker_v3.py`** | **Simple:** Minimalist, distraction-free | ProRes 4444 MOV (Alpha) |
| **`video_masker_v4.py`** | **Advanced:** HUD, instructions & status bar | ProRes 4444 MOV (Alpha) |

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

### 2. Install FFmpeg (Required for V3 & V4)
To generate transparent ProRes 4444 videos, you must have FFmpeg installed:
```bash
brew install ffmpeg
```

### 3. AI Model Checkpoint
1. Create a `checkpoints` folder.
2. Download `sam2_hiera_small.pt` from the official SAM 2 repository.
3. Place it inside the `checkpoints` folder.

---

## 📖 Usage Guide

### Using Video Workflows (V1, V2, V3, V4)
1. Export your clip from DaVinci Resolve into `/input_video`.
2. Run your preferred script (e.g., `python video_masker_v4.py`).
3. **Interaction:** 
    - **Left-Click (Green):** Select object.
    - **Right-Click (Red):** Exclude areas.
    - **Scrubber:** Move through time.
4. Adjust the **Feather** slider to soften mask edges.
5. Press **'P'** to propagate and export the result to `/output_video`.

### Using Image Sequence (`selector_v3.py`)
1. Export a PNG/JPEG sequence from DaVinci Resolve into `/raw_frames`.
2. Run `python selector_v3.py`.
3. The script automatically sanitizes filenames and converts PNGs to JPEGs for the AI.
4. Select your object and press **'P'**. Resulting masks appear in `/masked_output`.

---

## 🎨 DaVinci Resolve Integration

### For B&W Masks (V1, V2, and Image Sequence)
1. Drag the mask into your Media Pool and place it on the timeline above your footage.
2. In the **Fusion Page**, connect the mask to the **Blue (Effect Mask)** input of your footage node.
3. In the **Inspector -> Settings**:
    - Change **Channel** to **Luminance**.
    - Change **Mapping Mode** to **Stretch**.

### For Pro Cutouts (V3 and V4)
1. Drag the `cutout_xxxx.mov` file from `/output_video` directly onto your **Edit Page** timeline.
2. Place it on **Video Track 2** above your background.
3. **No Fusion required.** The transparency is built-in.

---

## ⌨️ Controls Summary
| Key | Action |
| :--- | :--- |
| **Left-Click** | Add selection point (Green) |
| **Right-Click** | Add exclusion point (Red) |
| **R** | Reset all selections (Advanced UI versions) |
| **P** | Start AI Processing & Export |
| **Q** | Quit Application |

---

## 📥 Download & Installation (macOS)

1. Go to the [Releases](link-to-your-release-page) page and download `AI_Masker_Pro.zip`.
2. Unzip the file and move `AI_Masker_Pro.app` to your **Applications** folder.

### ⚠️ Important: Security Note
Because this app is not signed by a registered Apple Developer, macOS will block it by default. 

**To open it:**
1. **Right-Click** the app and select **Open**.
2. A warning will appear; click **Open Anyway**.
3. You only need to do this once.

## 💻 Requirements
- macOS 13.0 or newer.
- Apple Silicon (M1, M2, M3).
- **FFmpeg** must be installed via Homebrew (`brew install ffmpeg`) if you want to use the ProRes export feature.
