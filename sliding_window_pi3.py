import os
import glob
import argparse
import numpy as np
import torch
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from demo_gradio import predictions_to_glb
from tqdm.auto import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Pi3 demo for large datasets using overlap-based alignment")
parser.add_argument("--data_path", type=str, default='/mnt/public/Ehsan/datasets/private/Najmeh/simulated_data/simulated_pandaset_V5_large/001/camera/front_camera',
                        help="Path to the input image directory or a video file.")
parser.add_argument("--save_path", type=str, default='./outputs/simulated_pandaset_V5_large/001/front_camera',
                    help="Path to save the output .ply file.")
parser.add_argument("--device", type=str, default='cuda',
                    help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")  
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--window_size", type=int, default=20, help="Number of frames per processing window")
parser.add_argument("--overlap", type=int, default=15, help="Number of overlapping frames between windows")
  
args = parser.parse_args()


def load_model():
    print(f"Loading model...")
    device = torch.device(args.device)
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    return model

def run_sliding_window(model, data_path, window_size, overlap):
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    all_img_paths = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
    total_frames = len(all_img_paths)

    # Calculate window positions
    stride = window_size - overlap
    window_starts = list(range(0, total_frames - window_size + 1, stride))
    
    # Ensure last window doesn't exceed bounds
    if window_starts[-1] + window_size < total_frames:
        window_starts.append(window_starts[-1]+stride)
        
    print('window_starts', window_starts)
    print(f"Will process {len(window_starts)} windows")

    all_poses = []
    # Process each window
    for window_idx, start_idx in enumerate(tqdm(window_starts, desc="Processing windows")):
        end_idx = min(start_idx + window_size, total_frames)
        window_image_paths = all_img_paths[start_idx:end_idx]
        
        print(f"\nWindow {window_idx + 1}/{len(window_starts)}: frames {start_idx}-{end_idx-1}")
        

        # Load and preprocess images for current window
        images = load_images_as_tensor(window_image_paths, interval=1).to(args.device) # (N, 3, H, W)
        print(f"Loaded window images shape: {images.shape}")
        
        # Run VGGT inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images[None])

        del predictions['local_points']
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

        # Handle alignment based on window
        if window_idx == 0:
            transformed_extrinsics = extrinsic_window
       
        if window_idx > 0:
            common_frame_idx = start_idx
            if common_frame_idx not in unified_results['extrinsics']:
                raise ValueError(f"Common frame {common_frame_idx} missing in previous window")

            prev_overlaps = []
            for k, v in unified_results['extrinsics'].items():
                if k >= common_frame_idx:
                    prev_overlaps.append(to_4x4(v.copy()))
            curr_overlaps = []
            for i in range(overlap):
                curr_overlaps.append(to_4x4(extrinsic_window[i].copy()))
            # for o_i in range(overlap-1):
            #     compare_relative_transforms(prev_overlaps[o_i], prev_overlaps[o_i+1], curr_overlaps[o_i], curr_overlaps[o_i+1])

            T_align = calculate_transformation(prev_overlaps, curr_overlaps)
            transformed_extrinsics = transform_poses_to_reference(
                extrinsic_window, T_align
            )
   
        for i in range(len(transformed_extrinsics)):
            global_frame_idx = start_idx + i
            # global_frame_idx = start_idx + i + (window_idx * overlap)
            # if global_frame_idx not in unified_results['extrinsics']:
            unified_results['extrinsics'][global_frame_idx] = transformed_extrinsics[i]
            unified_results['intrinsics'][global_frame_idx] = intrinsic_window[i]
            unified_results['images'][global_frame_idx] = images_np[i]
            unified_results['world_points'][global_frame_idx] = world_points_window[i]
            unified_results['world_points_conf'][global_frame_idx] = world_points_conf[i]
            unified_results['depth'][global_frame_idx] = depth_maps[i]
            unified_results['depth_conf'][global_frame_idx] = depth_confs[i]
            # else:
                # unified_results['extrinsics'][global_frame_idx] = extrinsic_averaging(unified_results['extrinsics'][global_frame_idx], 
                #                                                                       transformed_extrinsics[i])
                
        # Clear GPU memory
        del predictions, images
        torch.cuda.empty_cache()
        

    # Convert dictionaries to arrays (sorted by frame index)
    frame_indices = sorted(unified_results['extrinsics'].keys())
    
    final_results = {
        'images': np.stack([unified_results['images'][i] for i in frame_indices]),
        'extrinsic': np.stack([unified_results['extrinsics'][i] for i in frame_indices]),
        'intrinsic': np.stack([unified_results['intrinsics'][i] for i in frame_indices]),
        'world_points': np.stack([unified_results['world_points'][i] for i in frame_indices]),
        'world_points_conf': np.stack([unified_results['world_points_conf'][i] for i in frame_indices]),
        'depth': np.stack([unified_results['depth'][i] for i in frame_indices]),
        'depth_conf': np.stack([unified_results['depth_conf'][i] for i in frame_indices])
    }
    
    print(f"\nUnified processing complete with overlap-based alignment!")
    print(f"Final results shape:")
    print(f"- Images: {final_results['images'].shape}")
    print(f"- Extrinsics: {final_results['extrinsic'].shape}")
    print(f"- Intrinsics: {final_results['intrinsic'].shape}")
    print(f"- World points: {final_results['world_points'].shape}")
    print(f"- Depth: {final_results['depth'].shape}")
    
    return final_results


def main():
    model = load_model()
    os.makedirs(args.save_path, exist_ok=True)
    predictions = run_sliding_window(model, args.data_path, args.window_size, args.overlap)
    
    save_camera_poses_as_quaternion(predictions, os.path.join(args.save_path, "poses.json"))

    glbfile = os.path.join(
        args.save_path,
        f"glbscene.glb",
    )

    glbscene = predictions_to_glb(
        predictions,
        conf_thres=3.0,
        filter_by_frames="All",
        show_cam=True,
    )
    glbscene.export(file_obj=glbfile)


if __name__ == "__main__":
    main()