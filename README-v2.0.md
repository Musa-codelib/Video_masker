# Mk Masker V2.0 — AI Video Background Removal & Rotoscoping

Electron + Python desktop app for removing and rotoscoping video backgrounds using AI (SAM2 / RVM).

---

## Features

- **SAM2 Click-to-Track** — click on a subject in any frame, SAM2 propagates the mask across the entire video (Tiny and Small variants)
- **RVM Auto-Human** — Robust Video Matting automatically segments humans without any clicks
- **ProRes 4444 Export** — full-quality lossless output with alpha channel
- **H.265 Export** — smaller file size with balanced alpha quality
- **B&W Mask Export** — grayscale matte for compositing in other tools
- **Drag-and-Drop** — drop a video clip onto the main viewport to load it
- **Frame Scrubbing** — scrub through extracted frames with a timeline slider
- **Real-time Mask Preview** — see your selection as a translucent blue overlay before exporting
- **Setup Wizard** — first-launch wizard downloads AI model weights automatically

---

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.13+ with venv
- Node.js 18+
- ffmpeg (system install, for export)

---

## Project Structure

```
Mk_Masker_v2/
├── app/                    # Electron frontend
│   ├── index.html          # UI layout (inline styles)
│   ├── renderer.js         # Frontend logic (SocketIO, canvas, UI)
│   ├── main.js             # Electron main process (window, server launcher)
│   ├── package.json        # Electron + builder config
│   └── logo.png            # App icon
├── server/                 # Python backend
│   ├── server.py           # SocketIO server (aiohttp)
│   ├── core/
│   │   ├── sam2_runner.py  # SAM2 image annotator + video runner
│   │   └── rvm_runner.py   # Robust Video Matting runner
│   └── utils/
│       ├── video_handler.py # Frame extraction, base64 encoding
│       ├── export.py       # ProRes / H.265 / B&W export pipeline
│       ├── errors.py       # Error types and payloads
│       └── paths.py        # Checkpoint path resolution
├── venv/                   # Python virtual environment
├── checkpoints/            # AI model weights (SAM2, RVM)
└── bin/                    # System ffmpeg (user-installed)
```

---

## Development

### 1. Set up Python environment

```bash
source venv/bin/activate
pip install torch torchvision opencv-python pillow socketio aiohttp numpy
```

### 2. Install Node dependencies

```bash
cd app
npm install
```

### 3. Run

```bash
# Terminal 1 — start the Python server
cd server
python server.py

# Terminal 2 — start Electron
cd app
npm start
```

Or simply `npm start` from `app/` — `main.js` auto-launches the server.

---

## Architecture

```
┌──────────────┐     Socket.IO      ┌──────────────┐
│   Electron   │ ◄──────────────► │   Python     │
│   Renderer   │    localhost:8080  │   Server     │
│              │                    │              │
│  canvas      │  load_video       │  SAM2 / RVM  │
│  sidebar     │  add_click        │  extractor   │
│  drag-drop   │  start_processing │  exporter    │
└──────────────┘                    └──────────────┘
```

- **Renderer** handles all UI: drag-and-drop, canvas rendering, scrubbing, progress display
- **Server** handles all AI: frame extraction (OpenCV), model inference (SAM2/RVM), mask generation, video export (ffmpeg)
- Heavy CPU/GPU work runs in `run_in_executor` so the async event loop stays responsive

---

## Export Formats

| Format | Codec | Description |
|--------|-------|-------------|
| ProRes 4444 | `prores_videotoolbox` | Full quality, alpha channel preserved |
| H.265 | `hevc_videotoolbox` | Smaller file, balanced alpha quality |
| B&W Mask | Grayscale | Fusion-mode matte for compositing |

---

## License

Proprietary — Mk Masker Pro
