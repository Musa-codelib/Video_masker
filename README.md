# Mk Masker Pro (SAM 2 / RVM)

A professional-grade rotoscoping toolkit for DaVinci Resolve (Free/Studio). Powered by Meta's **Segment Anything Model 2 (SAM 2)** and **Robust Video Matting (RVM)**, optimized for **Apple Silicon (M1/M2/M3)**.

## 📥 Downloads (Standalone Apps)

*   🚀 **[Mk Masker Pro v2.0 (Latest)](https://github.com/Musa-codelib/Video_masker/releases/latest)** — *Unified Release: SAM 2 Small, Tiny, & RVM Support | Retro OS UI | Universal Platform Support.*
*   🧪 **[Mk Masker Lite v1.2](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.2-lite)** — *SAM 2.1 Tiny | Lightweight | Fast.*
*   🧪 **[Mk Masker Pro v1.2.0](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.2.0)** — *SAM 2 Small | Radiating Bundle Engine | 4K Stable.*
*   📎 **[Mk Masker Pro v1.1.0 (Legacy)](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.1.0)** — *Bi-Directional introduction.*
*   📎 **[AI Masker Pro v1.0 (Beta)](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.0-beta)** — *Initial release.*

---

## 🚀 Choose Your Workflow

### 1. Mk Masker Pro v2.0 (The Unified Experience)
**Best for:** All users. A complete overhaul featuring a retro-inspired UI and integrated model support.
- **Multi-Model Support:** Switch between SAM 2 (Small/Tiny) and RVM (Robust Video Matting) for human subjects.
- **Retro OS UI:** A nostalgic, high-performance interface.
- **Fit for All:** Optimized for stability across different workflows.

### 2. Mk Masker Lite v1.2 (Lightweight)
**Best for:** Users wanting a fast, lightweight tool.
- **SAM 2.1 Tiny** model — faster and smaller than the Pro build.
- **Radiating Bundle Engine** for infinite timeline support.
- **ProRes 4444** and **B&W Mask** modes.

### 3. Video-to-Video Workflows (Python Scripts)
| Script | Logic Style | Output |
| :--- | :--- | :--- |
| **`video_masker_v7.py`** | **Radiating Bundles (SAM 2.1 Tiny)** | ProRes 4444 or B&W Mask |

---

## 🛠️ Features & Stability

*   **Radiating Bundle Engine (RBE):** Automatically partitions video into 50-frame bundles with a "Hidden State" handshake. This eliminates the `MPSGraph INT_MAX` error, allowing for the processing of 4K and extremely long clips.
*   **Bi-Directional Tracking:** Select a "Hero Frame" anywhere in your clip; the AI tracks forward and backward simultaneously to cover the entire timeline.
*   **Zero-Grain Sync:** Implements strict GPU-to-CPU synchronization to ensure the selection mask is solid and free of digital noise.
*   **High-End Alpha:** Export ProRes 4444 videos with built-in transparency for "drag-and-drop" editing in the Resolve Edit Page.

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
- **FFmpeg** must be installed via Homebrew (`brew install ffmpeg`) for ProRes features (dev mode only; the standalone app bundles ffmpeg).
- For the **Lite** build: Place `sam2.1_hiera_tiny.pt` in `/checkpoints` and `sam2.1_hiera_t.yaml` in the repo root.
- For the **Pro** build: Place `sam2_hiera_small.pt` in `/checkpoints` and `sam2_hiera_s.yaml` in the repo root.

---

## 🎨 Mask Integration

### For Pro Cutouts (ProRes 4444)
Works for all editing softwares supporting ProRes 4444
1. Export your clip as an `.mp4` or `.mov`.
2. Run **Mk Masker Pro** and select **ProRes 4444** mode.
3. Drag the resulting `cutout_xxxx.mov` back into editing software.
4. Place it on **Track 2** above your background. **Transparency is automatic.**

### For B&W Masks (Fusion - DaVinci Resolve specific)
1. In the **Fusion Page**, connect the mask to the **Blue (Effect Mask)** input of your footage.
2. In the **Inspector -> Settings**: Change **Channel** to **Luminance** and **Mapping Mode** to **Stretch**.

---

## ⌨️ Controls Summary
| Key | Action |
| :--- | :--- |
| **Left-Click** | Add selection point (Green) |
| **Right-Click** | Add exclusion point (Red) |
| **R** | Reset all selections |
| **P** | Start Radiating AI Process & Export |
| **Q** | Quit Application |

---

### ⚠️ Security Note (macOS)
Because this app is independently developed, macOS will block it on first launch.
1. **Right-Click** `Mk Masker Lite v1.2.app` and select **Open**.
2. Click **Open Anyway** in the security popup.
