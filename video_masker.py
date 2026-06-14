import os
import cv2
import torch
import shutil
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

# State
frame_idx = 0
points_dict = {} 
labels_dict = {}

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, count

def extract_frames(video_path):
    print(f"🎬 Extracting frames from {video_path.name}...")
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    
    cap = cv2.VideoCapture(str(video_path))
    success, frame_count = True, 0
    while success:
        success, frame = cap.read()
        if success:
            # Save as high quality JPG for SAM 2
            frame_name = f"{frame_count:08d}.jpg"
            cv2.imwrite(str(TEMP_DIR / frame_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_count += 1
    cap.release()
    print(f"✅ Extracted {frame_count} frames.")
    return frame_count

def compile_video(output_path, fps, width, height, frame_names):
    print(f"📼 Compiling masked video at {fps} FPS...")
    # Use MP4V codec for Mac compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)
    
    # Check if masked_output exists, otherwise we use the ones in memory or disk
    # For now, let's assume we are saving frames to temp then compiling
    for frame_name in frame_names:
        mask_img = cv2.imread(str(TEMP_DIR / ("mask_" + frame_name)), cv2.IMREAD_GRAYSCALE)
        out.write(mask_img)
    
    out.release()
    print(f"✨ Masked video saved to: {output_path.name}")

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
    
    # 1. Detect Input Video
    video_files = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.mov"))
    if not video_files:
        print("❌ No video files found in input_video folder!")
        return
    
    target_video = video_files[0] # Process the first video found
    fps, width, height, total_frames = get_video_metadata(target_video)
    extract_frames(target_video)

    # 2. Load SAM 2
    print("🧠 Loading AI Brain...")
    predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT), device=device)
    inference_state = predictor.init_state(video_path=str(TEMP_DIR))
    
    frame_names = sorted([f.name for f in TEMP_DIR.glob("*.jpg")])
    
    cv2.namedWindow("AI Video Masker V2")
    cv2.createTrackbar("Frame", "AI Video Masker V2", 0, total_frames - 1, on_trackbar)
    cv2.setMouseCallback("AI Video Masker V2", click_event)

    print("\n--- CONTROLS ---")
    print("Slider: Scrubber | Left Click: Add Point | Right Click: Remove | 'P': Process | 'Q': Quit")

    while True:
        img_path = TEMP_DIR / frame_names[frame_idx]
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

        cv2.putText(display_img, f"Frame: {frame_idx}/{total_frames-1} | FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("AI Video Masker V2", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'):
            print("🚀 Propagating through video...")
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # Save temporary mask images to compile them
            for i, frame_name in enumerate(frame_names):
                mask_data = (video_segments[i][1].squeeze() * 255).astype(np.uint8)
                cv2.imwrite(str(TEMP_DIR / ("mask_" + frame_name)), mask_data)
            
            # 3. Compile the Final Video
            output_video_path = OUTPUT_DIR / f"mask_{target_video.stem}.mp4"
            compile_video(output_video_path, fps, width, height, frame_names)
            
            # Clean up temp frames to save space
            shutil.rmtree(TEMP_DIR)
            print("🧹 Temp frames cleaned. Process finished!")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()