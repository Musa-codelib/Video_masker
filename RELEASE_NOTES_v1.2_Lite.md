# Mk Masker Lite v1.2 — Release Notes

**Mk Roto Masker v1.2 Lite** — Powered by **SAM 2.1 Tiny**

---

## 🚀 What's New

### SAM 2.1 Tiny Model
- Upgraded from SAM 2 Small to **SAM 2.1 Tiny** — Meta's latest lightweight segmentation model.
- Smaller footprint, faster inference on Apple Silicon (MPS), and improved tracking quality over the original SAM 2.

### Lite Build, Full Power
- All the professional features of Mk Masker Pro, now running on the optimized Tiny architecture.
- Same **Radiating Bundle Engine** (RBE) — 50-frame chunked processing with hidden-state handoff.
- Same output modes: **ProRes 4444 (Alpha)** and **B&W Mask (Fusion)**.

### Performance
- Reduced model size: ~156 MB checkpoint vs ~375 MB (Small).
- Faster mask propagation on long clips.
- Lower GPU memory pressure — more headroom for 4K workflows.

### Stability Fixes
- **ffmpeg dylib fix:** Bundled ffmpeg now includes all required dynamic libraries, eliminating runtime SIGABRT crashes during ProRes export.
- **Silent failure fix:** ProRes encoding failures now surface as clear error messages instead of failing silently.
- **Improved instruction overlay:** Larger, more readable on-screen controls in the selector window.

---

## 📦 Downloads

| Asset | Description |
| :--- | :--- |
| **`Mk Masker Lite v1.2.app.zip`** | Standalone macOS app — no Python required. |
| `Source code (zip)` | Full source for this release. |
| `Source code (tar.gz)` | Full source for this release. |

---

## 🛠️ System Requirements

- **macOS 13.0 (Ventura) or later**
- **Apple Silicon** (M1 / M2 / M3 / M4)
- ~1.5 GB free disk space for the app bundle + model

---

## 🎨 Output Modes

### ProRes 4444 (Alpha)
- Exports a `cutout_xxxx.mov` with embedded transparency.
- Drag-and-drop into DaVinci Resolve, Final Cut Pro, or After Effects.
- Uses hardware-accelerated `prores_videotoolbox` encoder.

### B&W Mask (Fusion)
- Exports a `mask_xxxx.mp4` — pure white mask on black background.
- Use in DaVinci Resolve Fusion page as an Effect Mask (Channel: Luminance, Mapping Mode: Stretch).

---

## ⌨️ Controls

| Key | Action |
| :--- | :--- |
| **Left-Click** | Add selection point (Green) |
| **Right-Click** | Add exclusion point (Red) |
| **R** | Reset all selections |
| **P** | Start Radiating AI Process & Export |
| **Q** | Quit Application |

---

## ⚠️ Security Note (macOS)

Because this app is independently developed, macOS Gatekeeper will block it on first launch.

1. **Right-Click** `Mk Masker Lite v1.2.app` and select **Open**.
2. Click **Open Anyway** in the security popup.

---

## 📝 Full Changelog

### Added
- SAM 2.1 Tiny model support (`sam2.1_hiera_tiny.pt`).
- Larger instruction overlay with readable text in selector window.

### Changed
- Model checkpoint updated from `sam2_hiera_small.pt` → `sam2.1_hiera_tiny.pt`.
- Model config updated to `sam2.1_hiera_t.yaml`.

### Known Limitations
- Apple Silicon only (arm64). Intel Macs are not supported.
- macOS 13.0+ required for MPS and ProRes Videotoolbox support.
