import torch
import cv2
import sam2
from sam2.build_sam import build_sam2_video_predictor

def test_setup():
    print("--- Environment Check ---")
    
    # 1. Check PyTorch and MPS (Metal)
    print(f"PyTorch Version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) detected!")
        device = torch.device("mps")
    else:
        print("❌ MPS not detected. Using CPU (Will be slow).")
        device = torch.device("cpu")

    # 2. Check OpenCV
    print(f"OpenCV Version: {cv2.__version__}")

    # 3. Try to load the SAM 2 Model
    model_cfg = "sam2_hiera_s.yaml"
    sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
    
    try:
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        print("✅ SAM 2 Model loaded successfully onto GPU!")
    except Exception as e:
        print(f"❌ Error loading SAM 2: {e}")

if __name__ == "__main__":
    test_setup()