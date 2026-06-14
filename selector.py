import os
import cv2
import torch
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

# --- CONFIGURATION ---
RAW_DIR = "./raw_frames"
OUTPUT_DIR = "./masked_output"
CHECKPOINT = "./checkpoints/sam2_hiera_small.pt"
MODEL_CONFIG = "sam2_hiera_s.yaml"

# Setup Device
device = torch.device("mps")
torch.autocast(device_type="mps", enabled=True)

# Global variables for interaction
points = []
labels = [] # 1 for positive, 0 for negative

def click_event(event, x, y, flags, param):
    global points, labels
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print("Initializing SAM 2 Predictor...")
    predictor = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT, device=device)
    
    # Start tracking state
    inference_state = predictor.init_state(video_path=RAW_DIR)
    
    # Get list of images for display
    frame_names = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(('.png', '.jpg'))])
    first_frame_path = os.path.join(RAW_DIR, frame_names[0])
    first_frame = cv2.imread(first_frame_path)
    
    cv2.namedWindow("SAM 2 Selector")
    cv2.setMouseCallback("SAM 2 Selector", click_event)

    print("\n--- INSTRUCTIONS ---")
    print("Left Click:  Add object (Green)")
    print("Right Click: Remove area (Red)")
    print("'C':         Clear points")
    print("'P':         Process (Propagate) all frames")
    print("'Q':         Quit")

    while True:
        display_img = first_frame.copy()
        
        if len(points) > 0:
            # Get mask prediction for the current points
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=np.array(points, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32),
            )
            
            # Create blue overlay
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
            blue_mask = np.zeros_like(display_img)
            blue_mask[:] = [255, 0, 0] # Blue
            mask_overlay = cv2.bitwise_and(blue_mask, blue_mask, mask=mask)
            display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.6, 0)

            # Draw points
            for i, pt in enumerate(points):
                color = (0, 255, 0) if labels[i] == 1 else (0, 0, 255)
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, color, -1)

        cv2.imshow("SAM 2 Selector", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            points.clear()
            labels.clear()
            predictor.reset_state(inference_state)
        elif key == ord('p'):
            if len(points) == 0:
                print("Add at least one point first!")
                continue
            
            print("Processing sequence... please wait.")
            # Propagate through the video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Save results as B&W PNGs
            print(f"Saving masks to {OUTPUT_DIR}...")
            for i, frame_name in enumerate(frame_names):
                mask_data = video_segments[i][1].squeeze()
                # Convert True/False to 255/0
                binary_mask = (mask_data * 255).astype(np.uint8)
                out_path = os.path.join(OUTPUT_DIR, frame_name)
                cv2.imwrite(out_path, binary_mask)
            
            print("Done! Check the 'masked_output' folder.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()