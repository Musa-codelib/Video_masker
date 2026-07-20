import os, cv2, torch, shutil, time, subprocess, sys
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

CHUNK_SIZE = 50 

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    if relative_path == "ffmpeg":
        import shutil
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
    return os.path.join(os.path.abspath("."), relative_path)

def run_masker_engine(input_path, output_dir, mode):
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except: pass

    device = torch.device("mps")
    checkpoint = get_resource_path("checkpoints/sam2.1_hiera_tiny.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    ffmpeg_bin = get_resource_path("ffmpeg")

    temp_dir = Path(output_dir) / "_temp_mk_workspace"
    work_chunk_dir = Path(output_dir) / "_working_chunk"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # 1. Extraction
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(7))
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(str(temp_dir / f"{count:08d}.jpg"), frame)
        if mode == "prores": cv2.imwrite(str(temp_dir / f"orig_{count:08d}.png"), frame)
        count += 1
    cap.release()

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    inference_state = predictor.init_state(video_path=str(temp_dir))
    frame_names = sorted([f.name for f in temp_dir.glob("*.jpg")])

    # Interaction State
    st = {'f': 0, 'up': True, 'pts': {}, 'lbls': {}, 'msk': None, 'lf': -1}

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            st['pts'].setdefault(st['f'], []).append([x, y]); st['lbls'].setdefault(st['f'], []).append(1); st['up'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            st['pts'].setdefault(st['f'], []).append([x, y]); st['lbls'].setdefault(st['f'], []).append(0); st['up'] = True

    cv2.destroyAllWindows()
    win = "Mk Masker Selector"
    cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar("Frame", win, 0, total_frames - 1, lambda x: st.update({'f': x, 'up': True}))
    cv2.setMouseCallback(win, click)

    while True:
        if st['f'] != st['lf']:
            img = cv2.imread(str(temp_dir / frame_names[st['f']])); st['lf'] = st['f']
        disp = img.copy()
        if st['up']:
            if st['f'] in st['pts'] and st['pts'][st['f']]:
                with torch.inference_mode():
                    _, _, logits = predictor.add_new_points_or_box(inference_state, st['f'], 1, np.array(st['pts'][st['f']], dtype=np.float32), np.array(st['lbls'][st['f']], dtype=np.int32))
                st['msk'] = (logits[0, 0].cpu().numpy() > 0.0).astype(np.uint8) * 255
            else: st['msk'] = None
            st['up'] = False
        if st['msk'] is not None:
            ov = np.zeros_like(disp); ov[:, :, 0] = 255
            disp = cv2.addWeighted(disp, 1.0, cv2.bitwise_and(ov, ov, mask=st['msk']), 0.6, 0)
        
        overlay = disp.copy()
        cv2.rectangle(overlay, (5, 5), (700, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, disp, 0.25, 0, disp)
        cv2.putText(disp, "Mk Masker Lite v1.2", (18, 46), 1, 1.8, (255, 255, 255), 2)
        cv2.putText(disp, "L-Click: Add    R-Click: Remove", (18, 88), 1, 2, (0, 255, 0), 2)
        cv2.putText(disp, "'R': Reset    'P': Process    'Q': Quit", (18, 132), 1, 2, (200, 200, 200), 2)
        cv2.putText(disp, "SAM2.1 Tiny  |  MPS Accelerated  |  Apple Silicon", (18, 176), 1, 1.4, (0, 255, 255), 2)
        cv2.imshow(win, disp); k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('r'):
            st['pts'].clear(); st['lbls'].clear(); predictor.reset_state(inference_state)
            st['msk'], st['up'] = None, True
        elif k == ord('p') and st['pts']:
            all_segments = {}
            seed_frame = sorted(st['pts'].keys())[0]

            def run_partition(start_idx, end_idx, direction="forward", mask_handoff=None):
                if work_chunk_dir.exists(): shutil.rmtree(work_chunk_dir)
                work_chunk_dir.mkdir()
                for i in range(start_idx, end_idx): shutil.copy(str(temp_dir / f"{i:08d}.jpg"), work_chunk_dir)
                p_state = predictor.init_state(video_path=str(work_chunk_dir))
                for f_idx in st['pts']:
                    if start_idx <= f_idx < end_idx:
                        predictor.add_new_points_or_box(p_state, f_idx-start_idx, 1, np.array(st['pts'][f_idx], dtype=np.float32), np.array(st['lbls'][f_idx], dtype=np.int32))
                if mask_handoff is not None:
                    h_idx = 0 if direction == "forward" else (end_idx - start_idx - 1)
                    predictor.add_new_mask(p_state, h_idx, 1, mask_handoff)
                is_rev = (direction == "backward")
                for o_idx, _, o_logits in predictor.propagate_in_video(p_state, reverse=is_rev):
                    all_segments[start_idx + o_idx] = (o_logits[0, 0].cpu().numpy() > 0.0)
                handoff_idx = (end_idx - 1) if direction == "forward" else start_idx
                next_h = all_segments.get(handoff_idx)
                predictor.reset_state(p_state); torch.mps.empty_cache()
                return next_h

            # Radiate
            curr_h = None
            for s in range(seed_frame, total_frames, CHUNK_SIZE):
                curr_h = run_partition(s, min(s + CHUNK_SIZE, total_frames), "forward", curr_h)
            
            curr_h = all_segments.get(seed_frame)
            prev_s = seed_frame
            for s in range(seed_frame - CHUNK_SIZE, -CHUNK_SIZE, -CHUNK_SIZE):
                real_s = max(0, s)
                if real_s < prev_s:
                    curr_h = run_partition(real_s, prev_s, "backward", curr_h)
                    prev_s = real_s
            
            # Export
            blank = np.zeros((h, w), dtype=np.uint8)
            for i in range(total_frames):
                m = (all_segments[i] * 255).astype(np.uint8) if i in all_segments else blank
                if mode == "prores":
                    orig = cv2.imread(str(temp_dir / f"orig_{i:08d}.png"))
                    cv2.imwrite(str(temp_dir / f"rgba_{i:08d}.png"), cv2.merge([cv2.split(orig)[0], cv2.split(orig)[1], cv2.split(orig)[2], m]))
                else:
                    if i == 0: out_v = cv2.VideoWriter(str(Path(output_dir)/f"mask_{Path(input_path).stem}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), False)
                    out_v.write(m)

            if mode == "prores":
                output_path = Path(output_dir) / f"cutout_{Path(input_path).stem}.mov"
                try:
                    subprocess.run([ffmpeg_bin, '-y', '-framerate', str(fps), '-i', str(temp_dir/'rgba_%08d.png'), '-c:v', 'prores_videotoolbox', '-profile:v', '4', '-pix_fmt', 'ayuv64le', str(output_path)], check=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"FFmpeg ProRes encoding failed: {e}") from e
                if not output_path.exists():
                    raise RuntimeError(f"FFmpeg completed but output file not found: {output_path}")
            else:
                out_v.release()
            break
    cv2.destroyAllWindows(); shutil.rmtree(temp_dir); shutil.rmtree(work_chunk_dir, ignore_errors=True)