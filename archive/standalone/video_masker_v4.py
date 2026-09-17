import os
import cv2
import torch
import shutil
import time
import subprocess
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
BASE_DIR = Path("/Users/macbook/Desktop/code/AI_masker")
INPUT_DIR = BASE_DIR / "input_video"
OUTPUT_DIR = BASE_DIR / "output_video"
TEMP_DIR = BASE_DIR / "temp_frames"
CHECKPOINT = BASE_DIR / "checkpoints" / "sam2_hiera_small.pt"
MODEL_CONFIG = "sam2_hiera_s.yaml"

device = torch.device("mps")
torch.autocast(device_type="mps", enabled=True)

# State Management
frame_idx = 0
last_rendered_frame = -1
feather_val = 0
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, count

def extract_frames(video_path):
    """ Save JPG for AI and PNG for high-quality ProRes merge. """
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success: break
        cv2.imwrite(str(TEMP_DIR / f"{frame_count:08d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(TEMP_DIR / f"orig_{frame_count:08d}.png"), frame)
        frame_count += 1
    cap.release()
    return frame_count

def compile_prores(output_path, fps):
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(TEMP_DIR / 'rgba_%08d.png'),
        '-c:v', 'prores_videotoolbox', '-profile:v', '4', 
        '-pix_fmt', 'ayuv64le', # Best compatibility for M1 Alpha
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"❌ FFmpeg Error: {e}")

def set_status(text, color=(255, 255, 255), duration=3):
    global status_msg, status_color, status_expiry
    status_msg, status_color, status_expiry = text, color, time.time() + duration

def draw_ui(img, frame_count, feather):
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (380, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Frame: {frame_idx}/{frame_count-1}", (15, 30), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Feather: {feather}", (15, 55), font, 0.6, (0, 255, 255), 1)
    cv2.putText(img, "-"*35, (15, 75), font, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "Left Click: Add | Right Click: Remove", (15, 95), font, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "'R': Reset All | 'P': Process Cutout", (15, 125), font, 0.5, (200, 200, 200), 1)
    cv2.putText(img, "'Q': Quit", (15, 145), font, 0.5, (200, 200, 200), 1)
    cv2.putText(img, "Status:", (15, 190), font, 0.5, (200, 200, 200), 1)
    if time.time() < status_expiry:
        cv2.putText(img, status_msg, (80, 190), font, 0.5, status_color, 1)

def on_frame_trackbar(val):
    global frame_idx, needs_update
    frame_idx, needs_update = val, True

def on_feather_trackbar(val):
    global feather_val
    feather_val = val

def click_event(event, x, y, flags, param):
    global points_dict, labels_dict, frame_idx, needs_update
    if event == cv2.EVENT_LBUTTONDOWN:
        points_dict.setdefault(frame_idx, []).append([x, y]); labels_dict.setdefault(frame_idx, []).append(1)
        needs_update = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        points_dict.setdefault(frame_idx, []).append([x, y]); labels_dict.setdefault(frame_idx, []).append(0)
        needs_update = True

def main():
    global frame_idx, feather_val, points_dict, labels_dict, needs_update, cached_mask, last_rendered_frame
    
    video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.mov"))
    if not video_files: return print("❌ No video found!")
    
    target_video = video_files[0]
    fps, width, height, total_frames = get_video_metadata(target_video)
    extract_frames(target_video)

    predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT), device=device)
    inference_state = predictor.init_state(video_path=str(TEMP_DIR))
    frame_names = sorted([f.name for f in TEMP_DIR.glob("*.jpg")])
    
    cv2.namedWindow("AI Cutout Pro (ProRes 4444)")
    cv2.createTrackbar("Frame", "AI Cutout Pro (ProRes 4444)", 0, total_frames - 1, on_frame_trackbar)
    cv2.createTrackbar("Feather", "AI Cutout Pro (ProRes 4444)", 0, 50, on_feather_trackbar)
    cv2.setMouseCallback("AI Cutout Pro (ProRes 4444)", click_event)
    set_status("Ready", (0, 255, 0))

    while True:
        if frame_idx != last_rendered_frame:
            img_path = TEMP_DIR / f"orig_{frame_idx:08d}.png"
            frame_img = cv2.imread(str(img_path))
            last_rendered_frame = frame_idx
            needs_update = True

        display_img = frame_img.copy()
        
        if needs_update:
            if frame_idx in points_dict and len(points_dict[frame_idx]) > 0:
                _, _, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state, frame_idx=frame_idx, obj_id=1,
                    points=np.array(points_dict[frame_idx], dtype=np.float32),
                    labels=np.array(labels_dict[frame_idx], dtype=np.int32),
                )
                cached_mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
            else:
                cached_mask = None
            needs_update = False

        if cached_mask is not None:
            blue_m = np.zeros_like(display_img); blue_m[:] = [255, 0, 0]
            mask_overlay = cv2.bitwise_and(blue_m, blue_m, mask=cached_mask)
            display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.6, 0)
            for i, pt in enumerate(points_dict[frame_idx]):
                color = (0, 255, 0) if labels_dict[frame_idx][i] == 1 else (0, 0, 255)
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, color, -1)

        draw_ui(display_img, total_frames, feather_val)
        cv2.imshow("AI Cutout Pro (ProRes 4444)", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            points_dict.clear(); labels_dict.clear(); predictor.reset_state(inference_state)
            cached_mask, needs_update = None, True
            set_status("RESET ALL POINTS", (0, 255, 255))
        elif key == ord('p'):
            if not points_dict:
                set_status("ERROR: NO POINTS!", (0, 0, 255)); continue
            
            set_status("PROCESSING...", (0, 255, 0), duration=100)
            draw_ui(display_img, total_frames, feather_val)
            cv2.imshow("AI Cutout Pro (ProRes 4444)", display_img)
            cv2.waitKey(100)

            video_segments = {}
            for out_f_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_f_idx] = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
            
            print("🚀 Merging Alpha and Encoding ProRes...")
            for i in range(total_frames):
                orig = cv2.imread(str(TEMP_DIR / f"orig_{i:08d}.png"))
                mask = (video_segments[i] * 255)
                if feather_val > 0:
                    k = feather_val * 2 + 1
                    mask = cv2.GaussianBlur(mask, (k, k), 0)
                b, g, r = cv2.split(orig)
                cv2.imwrite(str(TEMP_DIR / f"rgba_{i:08d}.png"), cv2.merge([b, g, r, mask]))
            
            output_path = OUTPUT_DIR / f"cutout_{target_video.stem}.mov"
            compile_prores(output_path, fps)
            shutil.rmtree(TEMP_DIR)
            print("✨ Final Cutout Ready!")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()