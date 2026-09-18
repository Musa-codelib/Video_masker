# Mk Masker Pro (SAM 2 / RVM)

A professional-grade AI rotoscoping and background removal toolkit for video editors and compositors (DaVinci Resolve, Premiere Pro, After Effects, Nuke). Powered by Meta's **Segment Anything Model 2 (SAM 2)** and **Robust Video Matting (RVM)**, optimized for **Apple Silicon (M1/M2/M3/M4)** and Intel Macs.

---

## 📥 Downloads (Standalone Apps)

- 🚀 **[Mk Masker Pro v2.1 (Latest DMG)](https://github.com/Musa-codelib/Video_masker/releases/tag/v2.1)** — _Input & Output File Pickers | Bundled FFmpeg 8.1 | Non-Blocking AI Server & Frame Accuracy Fixes._
- 📦 **[Mk Masker Pro v2.0](https://github.com/Musa-codelib/Video_masker/releases/tag/v2.0)** — _Retro OS UI | Unified Engine: SAM 2 Small, Tiny & RVM Matting | ProRes 4444 Export | Socket.IO Desktop Architecture._
- 🧪 **[Mk Masker Lite v1.2](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.2-lite)** — _SAM 2.1 Tiny | Lightweight | Fast._
- 🧪 **[Mk Masker Pro v1.2.0](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.2.0)** — _SAM 2 Small | Radiating Bundle Engine | 4K Stable._
- 📎 **[Mk Masker Pro v1.1.0 (Legacy)](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.1.0)** — _Bi-Directional tracking introduction._
- 📎 **[AI Masker Pro v1.0 (Beta)](https://github.com/Musa-codelib/Video_masker/releases/tag/v1.0-beta)** — _Initial release._

---

## 🚀 Version Breakdown & Workflows

### 1. Mk Masker Pro v2.1 (Latest)

**Best for:** All production workflows requiring bulletproof standalone execution, high precision, and custom export output paths.

- 📂 **Input & Output Pickers:** Native file picker buttons for selecting video files and setting custom export folders (alongside drag-and-drop).
- 📦 **Bundled FFmpeg 8.1:** Includes all dynamic libraries inside the `.app` bundle — no Homebrew or system `ffmpeg` installation required.
- ⚡ **Async AI Server Engine:** AI initialization moved to non-blocking background executors; resolved `FRAME_MISSING` frame count mismatch bugs during OpenCV decoding.
- 🔧 **Environment-Aware Launcher:** Automatically locates Python virtual environments across packaged DMG, local development, and fallback paths.
- 🎨 **UI Polish:** Bug fixes and stability improvements to the Retro OS theme.

### 2. Mk Masker Pro v2.0 (Unified Baseline)

**Best for:** Core AI rotoscoping with dual-engine AI tracking.

- 🎨 **Retro OS Theme:** Win95 silver chrome panels, teal desktop background, phosphor-green status text, CRT-black viewport, and pixel-perfect monospace typography.
- 🤖 **Multi-Model Engine:** Toggle between **SAM 2 (Small/Tiny)** for point-and-click tracking or **RVM (Robust Video Matting)** for zero-click human segmentation.
- 🎬 **ProRes 4444 Export:** Native macOS `videotoolbox` hardware-accelerated encoding with alpha channel preservation.
- 🖥️ **Desktop Architecture:** Electron frontend communicating with an aiohttp/Socket.IO PyTorch backend.

---

## ⚡ Key Features

- **SAM 2 Interactive Tracking:** Click anywhere on an object/subject to generate high-fidelity spatial-temporal masks across complex video clips.
- **RVM Human Matting:** Automatic neural video matting specifically trained for human subject extraction without manual keyframing.
- **Hardware Acceleration:** Native PyTorch MPS (Metal Performance Shaders) execution on Apple Silicon for low-latency inference.
- **Flexible Export Modes:**
  - **Apple ProRes 4444:** Alpha channel output ready for DaVinci Resolve or Nuke.
  - **H.265 (HEVC with Alpha):** Compressed high-efficiency matte files.
  - **B&W Matte:** High-contrast black and white matte video streams for Fusion or After Effects track mattes.
- **Interactive Preview & Timeline Scrubbing:** Frame-by-frame timeline controls with real-time blue mask overlays.

---

## 🎬 Export Formats

| Format           | Codec                 | Alpha Channel  | Recommended Use                                        |
| :--------------- | :-------------------- | :------------: | :----------------------------------------------------- |
| **ProRes 4444**  | `prores_videotoolbox` |     ✅ Yes     | Final compositing, DaVinci Resolve, Nuke, Premiere Pro |
| **H.265 (HEVC)** | `hevc_videotoolbox`   |     ✅ Yes     | Web preview, lightweight storage                       |
| **B&W Matte**    | Grayscale H.264       | ❌ (RGB Matte) | Travel matte / Luma keyer input                        |

---

## 📄 License & Credits

- **License:** Proprietary — Mk Masker Pro
- **Models:** Powered by Meta AI's [Segment Anything 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2) and PeterL1n's [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting).
