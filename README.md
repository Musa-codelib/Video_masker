# Mk Masker Pro (SAM 2)
A professional-grade rotoscoping toolkit for **DaVinci Resolve (Free/Studio)**. This tool uses Meta's **Segment Anything Model 2 (SAM 2)** to generate high-quality masks, optimized for **Apple Silicon (M1/M2/M3)** via Metal (MPS) acceleration.

## 📥 Downloads (Standalone App)
The easiest way to use the tool. Download the bundled `.app` for macOS:

*   🚀 **[Mk Masker Pro v1.1.0 (Stable)](https://github.com/Musa-codelib/Video_masker/releases#release-v1.1.0)** — *New: Bi-Directional Tracking & 4K Stability Fix.*
*   🧪 **[Mk Masker Pro v1.0 (Beta)](https://github.com/Musa-codelib/Video_masker/releases#release-v1.0-beta)** — *Initial Release.*

---

## 🚀 Choose Your Workflow

### 1. Unified Standalone App (`Mk Masker Pro.app`)
**Best for:** Most users. A complete GUI-based application that handles everything from file selection to final export.
- Includes both **ProRes 4444** and **B&W Mask** modes.
- Branded interface with custom logo and icon.

### 2. Video-to-Video Workflows (Python Scripts)
| Script | Logic Style | Output |
| :--- | :--- | :--- |
| **`video_masker_v1.py`** | Simple UI | B&W MP4 Video |
| **`video_masker_v2.py`** | Advanced HUD | B&W MP4 Video |
| **`video_masker_v4.py`** | Advanced HUD | ProRes 4444 MOV (Alpha) |
| **`video_masker_v5.py`** | **Bi-Directional** | ProRes 4444 MOV (Alpha) |

### 3. Image Sequence Workflow
| Script | Use Case | Output |
| :--- | :--- | :--- |
| **`selector_v3.py`** | High-precision VFX / PNG Sequences | B&W JPEG Sequence |

---

## 🛠️ Features & Stability
*   **Bi-Directional Tracking:** Select a "Hero Frame" anywhere in your clip; the AI tracks forward and backward simultaneously.
*   **Stability First Engine:** Implements a smart 1024px internal scaling logic to prevent `MPSGraph INT_MAX` tensor overflows on high-resolution (4K) footage.
*   **Memory Offloading:** Efficiently manages 32GB+ of RAM to allow complex tracking on base-model MacBooks.
*   **High-End Alpha:** Export ProRes 4444 videos with built-in transparency for "drag-and-drop" editing.

---

## 📖 Installation (For Developers)

### 1. Clone & Environment
```bash
git clone https://github.com/Musa-codelib/Video_masker.git
cd Video_masker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. Requirements
- **macOS 13.0+** on **Apple Silicon**.
- **FFmpeg** must be installed via Homebrew (`brew install ffmpeg`) for ProRes features.
- Place `sam2_hiera_small.pt` in the `/checkpoints` folder.

---

## 🎨 DaVinci Resolve Integration

### For Pro Cutouts (ProRes 4444)
1. Drag the `cutout_xxxx.mov` file from your output folder directly onto your **Edit Page** timeline.
2. Place it on **Video Track 2** above your background.
3. **No Fusion required.** The transparency is built-in.

### For B&W Masks
1. In the **Fusion Page**, connect the mask to the **Blue (Effect Mask)** input of your footage.
2. In the **Inspector -> Settings**: Change **Channel** to **Luminance** and **Mapping Mode** to **Stretch**.

---

## ⌨️ Controls Summary
| Key | Action |
| :--- | :--- |
| **Left-Click** | Add selection point (Green) |
| **Right-Click** | Add exclusion point (Red) |
| **R** | Reset all selections |
| **P** | Start Bi-Directional AI Processing |
| **Q** | Quit Application |

---

### ⚠️ Security Note (macOS)
Because this app is not signed by an Apple Developer account, macOS will block it on first launch.
1. **Right-Click** `Mk Masker Pro.app` and select **Open**.
2. Click **Open Anyway** in the security popup.
