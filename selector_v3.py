import os
import cv2
import torch
import re
import shutil
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
BASE_DIR = Path("/Users/macbook/Desktop/code/AI_masker")
RAW_DIR = BASE_DIR / "raw_frames"
OUTPUT_DIR = BASE_DIR / "masked_output"
CHECKPOINT = BASE_DIR / "checkpoints" / "sam2_hiera_small.pt"
MODEL_CONFIG = "sam2_hiera_s.yaml"

device = torch.device("mps")
torch.autocast(device_type="mps", enabled=True)

# State for UI
frame_idx = 0
points_dict = {} 
labels_dict = {}

def prepare_workspace():
    """ Handles renaming, conversion, and cleaning old data. """
    print("--- 🛠️  Pre-Processing Started ---")
    
    # 1. Clean Masked Output (Janitor Mode)
    if OUTPUT_DIR.exists():
        print(f"Cleaning old masks in {OUTPUT_DIR.name}...")
        for f in OUTPUT_DIR.glob("*"):
            os.remove(f)
    else:
        OUTPUT_DIR.mkdir(parents=True)

    # 2. Find and Convert Files (Stripper & Converter Mode)
    files = list(RAW_DIR.glob("*"))
    if not files:
        print("❌ Error: No files found in raw_frames! Export PNGs from Resolve first.")
        return False

    print(f"Processing {len(files)} files...")
    
    for f_path in files:
        # Ignore hidden files like .DS_Store
        if f_path.name.startswith("."): continue
        
        # Regex to find the last group of numbers in the filename
        match = re.findall(r'(\d+)', f_path.stem)
        if match:
            new_name = f"{match[-1].zfill(8)}.jpg" # Pad to 8 digits for sorting
        else:
            # Fallback if no numbers found
            new_name = f_path.stem + ".jpg"

        target_path = RAW_DIR / new_name

        # Convert to JPG if it's a PNG or other format
        if f_path.suffix.lower() != ".jpg":
            img = cv2.imread(str(f_path))
            if img is not None:
                cv2.imwrite(str(target_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                os.remove(f_path) # Delete original PNG
                # print(f"Converted & Renamed: {f_path.name} -> {new_name}")
        
        # If it's already a JPG but has a messy name, just rename it
        elif f_path.name != new_name:
            os.rename(f_path, target_path)
            # print(f"Renamed: {f_path.name} -> {new_name}")

    print("✅ Pre-Processing Complete. Folder is clean.")
    return True

def on_trackbar(val):
    global frame_idx
    frame_idx = val

def click_event(event, x, y, flags, param):
    global points_dict, labels_dict, frame_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        points_dict.setdefault(frame_idx, []).append([x, y]); labels_dict.setdefault(frame_idx, []).append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points_dict.setdefault(frame_idx, []).append([x, y]); labels_dict.setdefault(frame_idx, []).append(0)

def main():
    global frame_idx
    
    # Run the Automation first
    if not prepare_workspace():
        return

    # Load SAM 2
    print("🧠 Loading AI Brain...")
    predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT), device=device)
    inference_state = predictor.init_state(video_path=str(RAW_DIR))
    
    # Get cleaned frames
    frame_names = sorted([f.name for f in RAW_DIR.glob("*.jpg")])
    num_frames = len(frame_names)
    
    cv2.namedWindow("AI Masker Pro")
    cv2.createTrackbar("Frame", "AI Masker Pro", 0, num_frames - 1, on_trackbar)
    cv2.setMouseCallback("AI Masker Pro", click_event)

    print("\n--- CONTROLS ---")
    print("Slider: Scrubber | Left Click: Add | Right Click: Remove | 'P': Process | 'Q': Quit")

    while True:
        img_path = RAW_DIR / frame_names[frame_idx]
        frame_img = cv2.imread(str(img_path))
        display_img = frame_img.copy()
        
        if frame_idx in points_dict and len(points_dict[frame_idx]) > 0:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=frame_idx, obj_id=1,
                points=np.array(points_dict[frame_idx], dtype=np.float32),
                labels=np.array(labels_dict[frame_idx], dtype=np.int32),
            )
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
            blue_mask = np.zeros_like(display_img); blue_mask[:] = [255, 0, 0]
            mask_overlay = cv2.bitwise_and(blue_mask, blue_mask, mask=mask)
            display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.6, 0)

            for i, pt in enumerate(points_dict[frame_idx]):
                color = (0, 255, 0) if labels_dict[frame_idx][i] == 1 else (0, 0, 255)
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, color, -1)

        cv2.putText(display_img, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("AI Masker Pro", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'):
            print("🚀 Tracking object through entire sequence...")
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            print(f"💾 Saving masks to {OUTPUT_DIR.name}...")
            for i, frame_name in enumerate(frame_names):
                mask_data = video_segments[i][1].squeeze()
                cv2.imwrite(str(OUTPUT_DIR / frame_name), (mask_data * 255).astype(np.uint8))
            print("✨ Done!")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()