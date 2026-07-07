import os, cv2, torch, shutil, time, subprocess, sys
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def run_masker_engine(input_path, output_dir, mode):
    # --- INTEL MAC FIX: Use CPU ---
    device = torch.device("cpu")
    
    # NOTE: On Intel CPUs, the 'small' model is slow. 
    # If it is too laggy, consider using 'sam2_hiera_tiny.pt' other is 'sam2_hiera_small.pt' 
    checkpoint = get_resource_path("checkpoints/sam2_hiera_tiny.pt")
    model_cfg = "sam2_hiera_s.yaml"
    ffmpeg_bin = get_resource_path("ffmpeg")

    temp_dir = Path(output_dir) / "_temp_mk_workspace"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # 1. Metadata & Extraction
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Scale calculation (Capped at 1024 for CPU efficiency)
    scale = min(1.0, 1024 / max(orig_w, orig_h))
    ai_w, ai_h = int(orig_w * scale), int(orig_h * scale)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        ai_frame = cv2.resize(frame, (ai_w, ai_h))
        cv2.imwrite(str(temp_dir / f"{count:08d}.jpg"), ai_frame)
        if mode == "prores": 
            cv2.imwrite(str(temp_dir / f"orig_{count:08d}.png"), frame)
        count += 1
    cap.release()

    # 2. Load SAM 2
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(video_path=str(temp_dir))
    
    frame_names = sorted([f.name for f in temp_dir.glob("*.jpg")])
    st = {'f': 0, 'up': True, 'pts': {}, 'lbls': {}, 'msk': None, 'lf': -1, 'fth': 0}

    def click(event, x, y, flags, param):
        ix, iy = int(x * scale), int(y * scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            st['pts'].setdefault(st['f'], []).append([ix, iy]); st['lbls'].setdefault(st['f'], []).append(1); st['up'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            st['pts'].setdefault(st['f'], []).append([ix, iy]); st['lbls'].setdefault(st['f'], []).append(0); st['up'] = True

    cv2.destroyAllWindows()
    win = "Mk Masker Selector (Intel CPU)"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar("Frame", win, 0, total_frames - 1, lambda x: st.update({'f': x, 'up': True}))
    cv2.createTrackbar("Feather", win, 0, 50, lambda x: st.update({'fth': x}))
    cv2.setMouseCallback(win, click)

    while True:
        if st['f'] != st['lf']:
            img_to_show = cv2.imread(str(temp_dir / frame_names[st['f']]))
            img_to_show = cv2.resize(img_to_show, (orig_w, orig_h))
            st['lf'] = st['f']
        
        display = img_to_show.copy()
        
        if st['up']:
            if st['f'] in st['pts'] and st['pts'][st['f']]:
                with torch.inference_mode():
                    _, _, logits = predictor.add_new_points_or_box(
                        inference_state, st['f'], 1, 
                        np.array(st['pts'][st['f']], dtype=np.float32), 
                        np.array(st['lbls'][st['f']], dtype=np.int32)
                    )
                mask_bool = (logits[0, 0] > 0.0).cpu().numpy()
                mask_uint8 = (mask_bool * 255).astype(np.uint8)
                st['msk'] = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            else: 
                st['msk'] = None
            st['up'] = False

        if st['msk'] is not None:
            ov = np.zeros_like(display)
            ov[:, :, 0] = 255 
            mask_vis = cv2.bitwise_and(ov, ov, mask=st['msk'])
            display = cv2.addWeighted(display, 1.0, mask_vis, 0.6, 0)
            for i, pt in enumerate(st['pts'][st['f']]):
                px, py = int(pt[0]/scale), int(pt[1]/scale)
                c = (0, 255, 0) if st['lbls'][st['f']][i] == 1 else (0, 0, 255)
                cv2.circle(display, (px, py), 5, c, -1)

        cv2.putText(display, f"Mk Pro | INTEL CPU | 'P' to Process", (15, 35), 1, 1.5, (255, 255, 255), 2)
        cv2.imshow(win, display)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('p') and st['pts']:
            segs = {}
            with torch.inference_mode():
                print("🚀 Processing Bi-Directional (CPU)...")
                for o_idx, _, o_logits in predictor.propagate_in_video(inference_state):
                    segs[o_idx] = (o_logits[0] > 0.0).cpu().numpy().astype(bool)
                for o_idx, _, o_logits in predictor.propagate_in_video(inference_state, reverse=True):
                    segs[o_idx] = (o_logits[0] > 0.0).cpu().numpy().astype(bool)
            
            print("💾 Compiling final frames...")
            blank_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for i in range(total_frames):
                if i in segs:
                    m_uint8 = (segs[i] * 255).astype(np.uint8).squeeze()
                    mask = cv2.resize(m_uint8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                else:
                    mask = blank_mask

                if st['fth'] > 0:
                    mask = cv2.GaussianBlur(mask, (st['fth']*2+1, st['fth']*2+1), 0)
                
                if mode == "prores":
                    orig = cv2.imread(str(temp_dir / f"orig_{i:08d}.png"))
                    cv2.imwrite(str(temp_dir / f"rgba_{i:08d}.png"), cv2.merge([cv2.split(orig)[0], cv2.split(orig)[1], cv2.split(orig)[2], mask]))
                else:
                    if i == 0:
                        out_v = cv2.VideoWriter(str(Path(output_dir)/f"mask_{Path(input_path).stem}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h), False)
                    out_v.write(mask)

            if mode == "prores":
                output_file = Path(output_dir) / f"cutout_{Path(input_path).stem}.mov"
                # --- INTEL MAC FIX: Use standard 'prores' software encoder ---
                subprocess.run([
                    ffmpeg_bin, '-y', '-framerate', str(fps), '-i', str(temp_dir/'rgba_%08d.png'), 
                    '-c:v', 'prores', '-profile:v', '4', '-pix_fmt', 'yuva444p10le', str(output_file)
                ])
            else:
                out_v.release()
            break

    cv2.destroyAllWindows(); shutil.rmtree(temp_dir)