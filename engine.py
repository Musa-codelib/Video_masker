import os, cv2, torch, shutil, time, subprocess, sys
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def run_masker_engine(input_path, output_dir, mode):
    # Mac Stability: Set start method for AI multiprocessing
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except: pass

    device = torch.device("mps")
    checkpoint = get_resource_path("checkpoints/sam2_hiera_small.pt")
    model_cfg = "sam2_hiera_s.yaml"
    ffmpeg_bin = get_resource_path("ffmpeg")

    temp_dir = Path(output_dir) / "_temp_ai_workspace"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # 1. Extraction
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(str(temp_dir / f"{count:08d}.jpg"), frame)
        if mode == "prores": cv2.imwrite(str(temp_dir / f"orig_{count:08d}.png"), frame)
        count += 1
    cap.release()

    # 2. AI Load
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(video_path=str(temp_dir))
    frame_names = sorted([f.name for f in temp_dir.glob("*.jpg")])

    # 3. Selector State
    st = {'f': 0, 'up': True, 'pts': {}, 'lbls': {}, 'msk': None, 'lf': -1, 'fth': 0}

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            st['pts'].setdefault(st['f'], []).append([x, y]); st['lbls'].setdefault(st['f'], []).append(1); st['up'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            st['pts'].setdefault(st['f'], []).append([x, y]); st['lbls'].setdefault(st['f'], []).append(0); st['up'] = True

    cv2.destroyAllWindows()
    win = f"AI Masker Selector"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar("Frame", win, 0, total_frames - 1, lambda x: st.update({'f': x, 'up': True}))
    cv2.createTrackbar("Feather", win, 0, 50, lambda x: st.update({'fth': x}))
    cv2.setMouseCallback(win, click)

    # 4. Interactive Loop
    while True:
        if st['f'] != st['lf']:
            img = cv2.imread(str(temp_dir / frame_names[st['f']]))
            st['lf'] = st['f']
        
        display = img.copy()
        if st['up']:
            if st['f'] in st['pts'] and st['pts'][st['f']]:
                _, _, logits = predictor.add_new_points_or_box(inference_state, st['f'], 1, np.array(st['pts'][st['f']], dtype=np.float32), np.array(st['lbls'][st['f']], dtype=np.int32))
                st['msk'] = (logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
            else: st['msk'] = None
            st['up'] = False

        if st['msk'] is not None:
            blue_m = np.zeros_like(display); blue_m[:] = [255, 0, 0]
            display = cv2.addWeighted(display, 1.0, cv2.bitwise_and(blue_m, blue_m, mask=st['msk']), 0.6, 0)
            for i, pt in enumerate(st['pts'][st['f']]):
                c = (0, 255, 0) if st['lbls'][st['f']][i] == 1 else (0, 0, 255)
                cv2.circle(display, (int(pt[0]), int(pt[1])), 5, c, -1)

        cv2.putText(display, f"Mode: {mode.upper()} | 'P' to Process", (15, 30), 1, 1.2, (255, 255, 255), 2)
        cv2.imshow(win, display)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('p') and st['pts']:
            # 5. Propagation
            segs = {}
            for o_idx, _, o_logits in predictor.propagate_in_video(inference_state):
                segs[o_idx] = (o_logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
            
            # 6. Export
            if mode == "prores":
                for i in range(total_frames):
                    orig = cv2.imread(str(temp_dir / f"orig_{i:08d}.png"))
                    m = (segs[i] * 255)
                    if st['fth'] > 0: m = cv2.GaussianBlur(m, (st['fth']*2+1, st['fth']*2+1), 0)
                    b, g, r = cv2.split(orig)
                    cv2.imwrite(str(temp_dir / f"rgba_{i:08d}.png"), cv2.merge([b, g, r, m.astype(np.uint8)]))
                
                output_file = Path(output_dir) / f"cutout_{Path(input_path).stem}.mov"
                subprocess.run([ffmpeg_bin, '-y', '-framerate', str(fps), '-i', str(temp_dir/'rgba_%08d.png'), 
                                '-c:v', 'prores_videotoolbox', '-profile:v', '4', '-pix_fmt', 'ayuv64le', str(output_file)])
            else:
                output_file = Path(output_dir) / f"mask_{Path(input_path).stem}.mp4"
                out_v = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), False)
                for i in range(total_frames):
                    m = (segs[i] * 255)
                    if st['fth'] > 0: m = cv2.GaussianBlur(m, (st['fth']*2+1, st['fth']*2+1), 0)
                    out_v.write(m.astype(np.uint8))
                out_v.release()
            break

    cv2.destroyAllWindows()
    shutil.rmtree(temp_dir)