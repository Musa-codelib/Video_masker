import os
import cv2
import torch
import shutil
import time
import subprocess
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION (v1.2-LITE / SAM2.1 tiny) ---
BASE_DIR = Path("/Users/macbook/Desktop/code/AI_masker")
INPUT_DIR = BASE_DIR / "input_video"
OUTPUT_DIR = BASE_DIR / "output_video"
TEMP_DIR = BASE_DIR / "temp_frames"
WORKING_CHUNK_DIR = BASE_DIR / "_working_chunk"
CHECKPOINT = BASE_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml" # Use the 2.1 specific path
CHUNK_SIZE = 50

device = torch.device("mps")
torch.autocast(device_type="mps", enabled=True)

# State Management
frame_idx = 0
last_rendered_frame = -1
points_dict = {}
labels_dict = {}
status_msg = "Ready"
status_color = (0, 255, 0)
status_expiry = 0
needs_update = True
cached_mask = None

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h, count = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    cap.release()
    return fps, w, h, count

def extract_frames(video_path):
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(str(TEMP_DIR / f"{count:08d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(TEMP_DIR / f"orig_{count:08d}.png"), frame)
        count += 1
    cap.release()
    return count

def compile_prores(output_path, fps):
    cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', str(TEMP_DIR / 'rgba_%08d.png'),
           '-c:v', 'prores_videotoolbox', '-profile:v', '4', '-pix_fmt', 'ayuv64le', str(output_path)]
    subprocess.run(cmd, check=True)

def set_status(text, color=(255, 255, 255), duration=3):
    global status_msg, status_color, status_expiry
    status_msg, status_color, status_expiry = text, color, time.time() + duration

def draw_ui(img, frame_count, f_idx):
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (420, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Frame: {f_idx}/{frame_count-1}", (15, 30), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Lite Radiating Engine v7 (SAM2.1 tiny)", (15, 55), font, 0.6, (0, 255, 255), 1)
    cv2.putText(img, "-"*40, (15, 75), font, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "L-Click: Add | R-Click: Remove", (15, 95), font, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "'R': Reset | 'P': Run Optimized Process", (15, 125), font, 0.5, (200, 200, 200), 1)
    cv2.putText(img, "Status:", (15, 160), font, 0.5, (200, 200, 200), 1)
    if time.time() < status_expiry:
        cv2.putText(img, status_msg, (80, 160), font, 0.5, status_color, 1)

def main():
    global frame_idx, points_dict, labels_dict, needs_update, cached_mask, last_rendered_frame

    if not CHECKPOINT.exists():
        return print(f"❌ Checkpoint missing: {CHECKPOINT}")

    video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.mov"))
    if not video_files: return print("❌ No video found!")

    target_video = video_files[0]
    fps, width, height, total_frames = get_video_metadata(target_video)
    extract_frames(target_video)

    predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT), device=device)
    inference_state = predictor.init_state(video_path=str(TEMP_DIR))

    win = "Mk Masker Lite Radiating Engine"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar("Frame", win, 0, total_frames - 1, lambda x: globals().update(frame_idx=x, needs_update=True))

    def click_event(event, x, y, flags, param):
        global needs_update
        if event == cv2.EVENT_LBUTTONDOWN:
            points_dict.setdefault(frame_idx, []).append([x, y]); labels_dict.setdefault(frame_idx, []).append(1); needs_update = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            points_dict.setdefault(frame_idx, []).append([x, y]); labels_dict.setdefault(frame_idx, []).append(0); needs_update = True

    cv2.setMouseCallback(win, click_event)
    set_status("Ready", (0, 255, 0))

    while True:
        if frame_idx != last_rendered_frame:
            img_path = TEMP_DIR / f"orig_{frame_idx:08d}.png"
            frame_img = cv2.imread(str(img_path))
            last_rendered_frame, needs_update = frame_idx, True

        display_img = frame_img.copy()

        if needs_update:
            if frame_idx in points_dict and points_dict[frame_idx]:
                with torch.inference_mode():
                    _, _, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state, frame_idx=frame_idx, obj_id=1,
                        points=np.array(points_dict[frame_idx], dtype=np.float32),
                        labels=np.array(labels_dict[frame_idx], dtype=np.int32),
                    )
                cached_mask = (out_mask_logits[0, 0].cpu().numpy() > 0.0).astype(np.uint8) * 255
            else: cached_mask = None
            needs_update = False

        if cached_mask is not None:
            ov = np.zeros_like(display_img); ov[:, :, 0] = 255
            display_img = cv2.addWeighted(display_img, 1.0, cv2.bitwise_and(ov, ov, mask=cached_mask), 0.6, 0)
            for i, pt in enumerate(points_dict.get(frame_idx, [])):
                c = (0, 255, 0) if labels_dict[frame_idx][i] == 1 else (0, 0, 255)
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, c, -1)

        draw_ui(display_img, total_frames, frame_idx)
        cv2.imshow(win, display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            points_dict.clear(); labels_dict.clear(); predictor.reset_state(inference_state)
            cached_mask = None; needs_update = True; set_status("RESET", (0, 255, 255))
        elif key == ord('p'):
            if not points_dict: continue
            set_status("PROCESSING...", (0, 255, 0), duration=1000)
            cv2.waitKey(10)

            all_segments = {}
            seed_frame = sorted(points_dict.keys())[0]

            def run_partition(start_idx, end_idx, direction="forward", mask_handoff=None):
                if WORKING_CHUNK_DIR.exists(): shutil.rmtree(WORKING_CHUNK_DIR)
                WORKING_CHUNK_DIR.mkdir()
                for i in range(start_idx, end_idx):
                    shutil.copy(str(TEMP_DIR / f"{i:08d}.jpg"), WORKING_CHUNK_DIR)

                p_state = predictor.init_state(video_path=str(WORKING_CHUNK_DIR))

                for f_idx in points_dict:
                    if start_idx <= f_idx < end_idx:
                        predictor.add_new_points_or_box(p_state, f_idx - start_idx, 1,
                            np.array(points_dict[f_idx], dtype=np.float32),
                            np.array(labels_dict[f_idx], dtype=np.int32))

                if mask_handoff is not None:
                    h_idx = 0 if direction == "forward" else (end_idx - start_idx - 1)
                    predictor.add_new_mask(p_state, h_idx, 1, mask_handoff)

                is_rev = (direction == "backward")
                for o_idx, _, o_logits in predictor.propagate_in_video(p_state, reverse=is_rev):
                    all_segments[start_idx + o_idx] = (o_logits[0, 0].cpu().numpy() > 0.0)

                handoff_idx = (end_idx - 1) if direction == "forward" else start_idx
                next_h = all_segments.get(handoff_idx)
                predictor.reset_state(p_state)
                del p_state
                torch.mps.empty_cache()
                return next_h

            # --- FORWARD CHAIN ---
            curr_h = None
            for s in range(seed_frame, total_frames, CHUNK_SIZE):
                e = min(s + CHUNK_SIZE, total_frames)
                print(f"📦 Forward Chunk: {s} to {e}")
                curr_h = run_partition(s, e, "forward", curr_h)

            # --- BACKWARD CHAIN (OPTIMIZED) ---
            curr_h = all_segments.get(seed_frame)
            prev_start = seed_frame

            for s in range(seed_frame - CHUNK_SIZE, -CHUNK_SIZE, -CHUNK_SIZE):
                real_s = max(0, s)
                real_e = prev_start

                if real_s < real_e:
                    print(f"📦 Backward Chunk: {real_s} to {real_e}")
                    curr_h = run_partition(real_s, real_e, "backward", curr_h)
                    prev_start = real_s

            # --- EXPORT ---
            print("💾 Compiling Final Video...")
            blank_mask = np.zeros((height, width), dtype=np.uint8)
            for i in range(total_frames):
                mask = (all_segments[i] * 255).astype(np.uint8) if i in all_segments else blank_mask
                orig = cv2.imread(str(TEMP_DIR / f"orig_{i:08d}.png"))
                cv2.imwrite(str(TEMP_DIR / f"rgba_{i:08d}.png"), cv2.merge([cv2.split(orig)[0], cv2.split(orig)[1], cv2.split(orig)[2], mask]))

            compile_prores(OUTPUT_DIR / f"cutout_{target_video.stem}.mov", fps)
            if WORKING_CHUNK_DIR.exists(): shutil.rmtree(WORKING_CHUNK_DIR)
            shutil.rmtree(TEMP_DIR)
            print("✨ Done!")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
