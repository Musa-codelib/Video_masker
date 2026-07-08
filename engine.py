import os, cv2, torch, shutil, time, subprocess, sys
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def run_masker_engine(input_path, output_dir, mode):
    # 1. SETUP & PATHS
    device = torch.device("mps")
    checkpoint = get_resource_path("checkpoints/sam2_hiera_small.pt")
    model_cfg = "sam2_hiera_s.yaml"
    ffmpeg_bin = get_resource_path("ffmpeg")

    temp_dir = Path(output_dir) / "_temp_mk_workspace"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # 2. METADATA & HYBRID EXTRACTION
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate scale to cap AI at 1024px (Prevents INT_MAX overflow)
    scale = min(1.0, 1024 / max(orig_w, orig_h))
    ai_w, ai_h = int(orig_w * scale), int(orig_h * scale)

    print(f"🎬 Extracting {total_frames} frames. AI working resolution: {ai_w}x{ai_h}")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Save SMALL JPG for the AI (Stability)
        ai_frame = cv2.resize(frame, (ai_w, ai_h))
        cv2.imwrite(str(temp_dir / f"{count:08d}.jpg"), ai_frame)
        # Save FULL PNG for the high-end ProRes merge (Quality)
        if mode == "prores": 
            cv2.imwrite(str(temp_dir / f"orig_{count:08d}.png"), frame)
        count += 1
    cap.release()

    # 3. LOAD AI WITH MEMORY OFFLOADING
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(
        video_path=str(temp_dir),
        offload_video_to_cpu=True, # Prevent GPU Memory Bloat
        offload_state_to_cpu=True
    )
    
    frame_names = sorted([f.name for f in temp_dir.glob("*.jpg")])
    st = {'f': 0, 'up': True, 'pts': {}, 'lbls': {}, 'msk': None, 'lf': -1, 'fth': 0}

    def click(event, x, y, flags, param):
        # Map High-Res UI clicks to Low-Res AI coordinates
        ix, iy = int(x * scale), int(y * scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            st['pts'].setdefault(st['f'], []).append([ix, iy]); st['lbls'].setdefault(st['f'], []).append(1); st['up'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            st['pts'].setdefault(st['f'], []).append([ix, iy]); st['lbls'].setdefault(st['f'], []).append(0); st['up'] = True

    cv2.destroyAllWindows()
    win = "Mk Masker Selector"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar("Frame", win, 0, total_frames - 1, lambda x: st.update({'f': x, 'up': True}))
    cv2.createTrackbar("Feather", win, 0, 50, lambda x: st.update({'fth': x}))
    cv2.setMouseCallback(win, click)

    # 4. INTERACTIVE LOOP
    while True:
        if st['f'] != st['lf']:
            # Load the small frame but show it large for the user
            img = cv2.imread(str(temp_dir / frame_names[st['f']]))
            img_disp = cv2.resize(img, (orig_w, orig_h))
            st['lf'] = st['f']
        
        display = img_disp.copy()
        
        if st['up']:
            if st['f'] in st['pts'] and st['pts'][st['f']]:
                with torch.inference_mode():
                    torch.mps.synchronize() # Wait for previous GPU tasks
                    _, _, logits = predictor.add_new_points_or_box(
                        inference_state, st['f'], 1, 
                        np.array(st['pts'][st['f']], dtype=np.float32), 
                        np.array(st['lbls'][st['f']], dtype=np.int32)
                    )
                    # Convert to clean uint8 on GPU before moving to CPU
                    mask_gpu = (logits[0, 0] > 0.0).to(torch.uint8) * 255
                    torch.mps.synchronize()
                    m_cpu = mask_gpu.cpu().numpy()
                    st['msk'] = cv2.resize(m_cpu, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            else: 
                st['msk'] = None
            st['up'] = False

        if st['msk'] is not None:
            ov = np.zeros_like(display); ov[:, :, 0] = 255 
            display = cv2.addWeighted(display, 1.0, cv2.bitwise_and(ov, ov, mask=st['msk']), 0.6, 0)
            for i, pt in enumerate(st['pts'][st['f']]):
                px, py = int(pt[0]/scale), int(pt[1]/scale)
                c = (0, 255, 0) if st['lbls'][st['f']][i] == 1 else (0, 0, 255)
                cv2.circle(display, (px, py), 5, c, -1)

        cv2.putText(display, f"Mk Pro | Hybrid Engine v5.6 | Frame {st['f']}", (15, 35), 1, 1.5, (255, 255, 255), 2)
        cv2.imshow(win, display)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('p') and st['pts']:
            # 5. BI-DIRECTIONAL PROCESSING
            torch.mps.empty_cache()
            segs = {}
            print("🚀 Processing Bi-Directional High-Res sequence...")
            with torch.inference_mode():
                for o_idx, _, o_logits in predictor.propagate_in_video(inference_state):
                    segs[o_idx] = (o_logits[0, 0] > 0.0).cpu().numpy()
                for o_idx, _, o_logits in predictor.propagate_in_video(inference_state, reverse=True):
                    segs[o_idx] = (o_logits[0, 0] > 0.0).cpu().numpy()
            
            # 6. QUALITY RECONSTRUCTION
            blank_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for i in range(total_frames):
                if i in segs:
                    m_u8 = (segs[i] * 255).astype(np.uint8)
                    mask = cv2.resize(m_u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
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
                subprocess.run([ffmpeg_bin, '-y', '-framerate', str(fps), '-i', str(temp_dir/'rgba_%08d.png'), 
                                '-c:v', 'prores_videotoolbox', '-profile:v', '4', '-pix_fmt', 'ayuv64le', str(output_file)])
            else:
                out_v.release()
            break

    cv2.destroyAllWindows(); shutil.rmtree(temp_dir)