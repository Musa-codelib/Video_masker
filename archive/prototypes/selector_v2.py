import os
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
RAW_DIR = "./raw_frames"
OUTPUT_DIR = "./masked_output"
CHECKPOINT = "./checkpoints/sam2_hiera_small.pt"
MODEL_CONFIG = "sam2_hiera_s.yaml"

device = torch.device("mps")
torch.autocast(device_type="mps", enabled=True)

# State Management
frame_idx = 0
points_dict = {} # frame_idx -> list of points
labels_dict = {} # frame_idx -> list of labels

def on_trackbar(val):
    global frame_idx
    frame_idx = val

def click_event(event, x, y, flags, param):
    global points_dict, labels_dict, frame_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        points_dict.setdefault(frame_idx, []).append([x, y])
        labels_dict.setdefault(frame_idx, []).append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points_dict.setdefault(frame_idx, []).append([x, y])
        labels_dict.setdefault(frame_idx, []).append(0)

def main():
    global frame_idx
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    predictor = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT, device=device)
    inference_state = predictor.init_state(video_path=RAW_DIR)
    
    frame_names = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(('.jpg', '.jpeg'))])
    num_frames = len(frame_names)
    
    cv2.namedWindow("AI Selector Pro")
    cv2.createTrackbar("Frame", "AI Selector Pro", 0, num_frames - 1, on_trackbar)
    cv2.setMouseCallback("AI Selector Pro", click_event)

    print("\n--- NEW CONTROLS ---")
    print("Slider:      Scrub through frames")
    print("Left Click:  Add point (Green)")
    print("Right Click: Remove point (Red)")
    print("'C':         Clear points on CURRENT frame")
    print("'P':         Process entire sequence")
    print("'Q':         Quit")

    while True:
        # Load current frame
        img_path = os.path.join(RAW_DIR, frame_names[frame_idx])
        frame_img = cv2.imread(img_path)
        display_img = frame_img.copy()
        
        # If we have points on this frame, show the preview
        if frame_idx in points_dict and len(points_dict[frame_idx]) > 0:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
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

        cv2.putText(display_img, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("AI Selector Pro", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            points_dict[frame_idx] = []
            labels_dict[frame_idx] = []
            predictor.reset_state(inference_state) # Note: simpler to reset all for now in this version
        elif key == ord('p'):
            print("Propagating and saving masks...")
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            for i, frame_name in enumerate(frame_names):
                mask_data = video_segments[i][1].squeeze()
                binary_mask = (mask_data * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(OUTPUT_DIR, frame_name), binary_mask)
            print("Sequence saved to masked_output!")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()