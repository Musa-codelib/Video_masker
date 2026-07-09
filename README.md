# Mk Masker Pro (SAM 2)
A professional-grade standalone rotoscoping/masking application. Powered by Meta's **Segment Anything Model 2 (SAM 2)** and optimized specifically for **Apple Silicon (M1/M2/M3)** via Metal (MPS) acceleration.

## 📥 Downloads (Standalone App)
The easiest way to use the tool. Download the bundled `.app` for macOS Silicon:

*   🚀 **[Mk Masker Pro v1.2.0 (Stable)](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.2.0)** — *Latest: Radiating Bundle Engine & 4K stability fix.*
*   🧪 **[Mk Masker Pro v1.1.0 (Legacy)](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.1.0)** — *Bi-Directional introduction.*
*   📎 **[Mk Masker Pro v1.0 (Beta)](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.0.0)** — *Initial Release.*

---

## 🚀 Choose Your Workflow

### 1. Unified Standalone App (`Mk Masker Pro 1.2.app`)
**Best for:** Most users. A complete GUI-based application that handles everything from file selection to final export.
- Includes **ProRes 4444** and **B&W Mask** modes.
- Powered by the **Radiating Bundle Engine** for infinite timeline support.
- Branded interface with native macOS icons.

### 2. Video-to-Video Workflows (Python Scripts)
| Script | Logic Style | Output |
| :--- | :--- | :--- |
| **`video_masker_v6.py`** | **Radiating Bundles** | ProRes 4444 or B&W Mask |
| **`video_masker_v5.py`** | Bi-Directional | ProRes 4444 MOV (Alpha) |
| **`video_masker_v1.py`** | Simple Pass | B&W MP4 Video |

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
- **FFmpeg** must be installed via Homebrew (`brew install ffmpeg`) for ProRes features.
- Place `sam2_hiera_small.pt` in the `/checkpoints` folder.

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
1. **Right-Click** `Mk Masker Pro 1.2.app` and select **Open**.
2. Click **Open Anyway** in the security popup.
