import argparse
import cv2
import numpy as np
import os
import torch
import matplotlib
import logging
from tqdm import tqdm
from PIL import Image

# --- THE CORRECT IMPORT ---
from depth_anything_3.api import DepthAnything3

# Silence logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("depth_anything_3").setLevel(logging.WARNING)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V3 Video Generator')
    parser.add_argument('--video-path', type=str, required=True, help='Path to video file')
    parser.add_argument('--outdir', type=str, default='./vis_video_da3')
    parser.add_argument('--encoder', type=str, default='da3-base', 
                        choices=['da3-small', 'da3-base', 'da3-large'],
                        help='Model size.')
    parser.add_argument('--pred-only', action='store_true', help='Only save the depth map')
    parser.add_argument('--grayscale', action='store_true', help='Save in grayscale')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading Model: depth-anything/{args.encoder} on {DEVICE}...")
    try:
        model = DepthAnything3.from_pretrained(f"depth-anything/{args.encoder}").to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    if not os.path.isfile(args.video_path):
        print(f"Video file not found: {args.video_path}")
        exit()

    os.makedirs(args.outdir, exist_ok=True)
    filename = args.video_path
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    print(f"Processing: {filename}")
    raw_video = cv2.VideoCapture(filename)
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

    output_width = frame_width if args.pred_only else frame_width * 2 + 50
    out_name = os.path.splitext(os.path.basename(filename))[0] + f'_{args.encoder}.mp4'
    output_path = os.path.join(args.outdir, out_name)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
    
    with tqdm(total=total_frames, unit="frame") as pbar:
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            try:
                # 1. Convert BGR (OpenCV) to RGB
                image_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                
                # 2. Convert to PIL Image (DA3 API likes this best)
                pil_image = Image.fromarray(image_rgb)
                
                # 3. Run Inference
                # Pass as a LIST: [pil_image]
                prediction = model.inference([pil_image])
                
                # 4. Extract Depth
                # prediction.depth is [N, H, W] -> [1, H, W]
                # We need to squeeze it to [H, W]
                depth = prediction.depth
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                
                # Squeeze the batch dimension (removes the '1' at the start)
                depth = np.squeeze(depth)
                
                # 5. Normalize (Robust min/max)
                d_min, d_max = depth.min(), depth.max()
                if d_max > d_min:
                    depth_norm = (depth - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(depth)

                # 6. Colorize
                if args.grayscale:
                    depth_vis = (depth_norm * 255).astype(np.uint8)
                    depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
                else:
                    depth_vis = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
                    depth_vis = depth_vis[:, :, ::-1] # RGB to BGR for VideoWriter

                # 7. Resize (DA3 output might not match input exactly due to padding)
                if depth_vis.shape[:2] != (frame_height, frame_width):
                    depth_vis = cv2.resize(depth_vis, (frame_width, frame_height))

                # 8. Write
                if args.pred_only:
                    out.write(depth_vis)
                else:
                    sep = np.ones((frame_height, 50, 3), dtype=np.uint8) * 255
                    combined = cv2.hconcat([raw_frame, sep, depth_vis])
                    out.write(combined)
            
            except Exception as e:
                # Use repr() to get more info if e is weird
                print(f"Frame Error: {repr(e)}")
                # If we fail, break so we don't spam errors
                break
            
            pbar.update(1)
    
    raw_video.release()
    out.release()
    print(f"Done! Saved to: {output_path}")