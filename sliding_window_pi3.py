import os
import glob
import argparse
import numpy as np
import torch
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from demo_gradio import predictions_to_glb
from tqdm.auto import tqdm
import json 
from scipy.spatial.transform import Rotation as R
from pi3.utils.geometry import depth_edge
import time
import viser
from scipy.optimize import least_squares

# Argument parser
parser = argparse.ArgumentParser(description="Pi3 demo for large datasets using overlap-based alignment")
parser.add_argument("--data_path", type=str, default='/mnt/public/Ehsan/datasets/private/Najmeh/simulated_data/simulated_pandaset_V5_pi3_estimated/001/camera/front_camera',
                        help="Path to the input image directory or a video file.")
parser.add_argument("--save_path", type=str, default='/mnt/public/Ehsan/datasets/private/Najmeh/simulated_data/simulated_pandaset_V5_pi3_estimated/001/camera/front_camera',
                    help="Path to save the output .ply file.")
parser.add_argument("--device", type=str, default='cuda',
                    help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")  
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--window_size", type=int, default=50, help="Number of frames per processing window")
parser.add_argument("--overlap", type=int, default=20, help="Number of overlapping frames between windows")
  
args = parser.parse_args()


def load_model():
    print(f"Loading model...")
    device = torch.device(args.device)
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    return model

def umeyama_sim3(src, dst):
    """
    Compute similarity transform (Sim(3)) from src → dst.
    Solves for s * R * x + t = y

    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points

    Returns:
        s: scale (float)
        R: rotation matrix (3x3)
        t: translation vector (3,)
    """
    assert src.shape == dst.shape, "Shape mismatch"
    N = src.shape[0]

    # Means
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # Centered vectors
    X = src - mu_src
    Y = dst - mu_dst

    # Covariance matrix
    cov = X.T @ Y / N

    # SVD
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1

    # Rotation
    R = U @ S @ Vt

    # Scale
    var_src = np.sum(X ** 2) / N
    s = np.trace(np.diag(D) @ S) / var_src

    # Translation
    t = mu_dst - s * R @ mu_src

    return s, R, t



def calculate_transformation(prev_overlaps, curr_overlaps):
    """
    Calculate noise-free transformation between two windows
    Args:
        prev_overlaps (np.ndarray): Previous overlaps of shape (S, 3, 4)
        curr_overlaps (np.ndarray): Current overlaps of shape (S, 3, 4)
    Returns:
        np.ndarray: noise-free transformation of shape (3, 4)
    """
    # camera centres (world coords) are rows 3 of inv(E)
    S, R, t = umeyama_sim3(curr_overlaps[:, :3, 3], prev_overlaps[:, :3, 3])
  

    return S, R, t



def transform_poses_sim3(poses, s, R_align, t_align):
    """
    Apply Sim(3) to a batch of camera-to-world poses.
    Args:
        poses: (N, 4, 4) camera-to-world poses
        s: scale
        R_align: (3, 3) rotation matrix
        t_align: (3,) translation vector

    Returns:
        Transformed poses: (N, 4, 4)
    """
    transformed = []
    for pose in poses:
        R_pose = pose[:3, :3]
        t_pose = pose[:3, 3]

        # Apply sim(3) transform: R' = R_align * R_pose, t' = s * R_align * t_pose + t_align
        R_new = R_align @ R_pose
        t_new = s * (R_align @ t_pose) + t_align

        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = t_new
        transformed.append(T_new)

    return np.stack(transformed)


def visualize_camera_poses(poses, scale=0.1):
    """
    Visualize camera-to-world poses in Viser.
    Each pose is shown as a coordinate frame or a frustum.
    """
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    time.sleep(1.0)  # Wait a second for server to initialize

    for i, pose in enumerate(poses):
        # Extract rotation and translation
        R_world = pose[:3, :3]
        t_world = pose[:3, 3]

        # Convert rotation matrix to quaternion (Viser expects w, x, y, z)
        quat = R.from_matrix(R_world).as_quat()  # x, y, z, w
        quat = [quat[3], quat[0], quat[1], quat[2]]

        server.scene.add_frame(
            name=f"camera_{i}",
            wxyz=np.array(quat),
            position=np.array(t_world),
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
             )

    print("View at: http://localhost:8080")
    # server.run_forever()
    while True:
        time.sleep(1.0)


def debug_alignment(prev_poses, curr_poses_raw, curr_poses_aligned):
    for i in range(len(prev_poses)):
        p = prev_poses[i][:3, 3]
        c_raw = curr_poses_raw[i][:3, 3]
        c_aligned = curr_poses_aligned[i][:3, 3]

        pos_error = np.linalg.norm(c_aligned - p)

        R_prev = R.from_matrix(prev_poses[i][:3, :3])
        R_aligned = R.from_matrix(curr_poses_aligned[i][:3, :3])
        rot_error_deg = R_prev.inv() * R_aligned
        angle_deg = rot_error_deg.magnitude() * 180 / np.pi

        print(f"[{i}] Pos error: {pos_error:.4f} m | Rot error: {angle_deg:.2f}°")


def sim3_residual(params, src_poses, tgt_poses):
    """
    Compute residuals between Sim(3)-aligned source and target poses.

    Params:
        params: [rotvec (3), trans (3), scale (1)]
        src_poses, tgt_poses: (N, 4, 4)

    Returns:
        residuals: (N * 6,)
    """
    rotvec = params[0:3]
    t = params[3:6]
    scale = params[6]

    R_opt = R.from_rotvec(rotvec).as_matrix()

    residuals = []
    for src, tgt in zip(src_poses, tgt_poses):
        R_src, t_src = src[:3, :3], src[:3, 3]
        R_tgt, t_tgt = tgt[:3, :3], tgt[:3, 3]

        R_aligned = R_opt @ R_src
        t_aligned = scale * (R_opt @ t_src) + t

        # Rotation residual
        dR = R.from_matrix(R_tgt.T @ R_aligned)
        rot_error = dR.as_rotvec()

        # Translation residual
        trans_error = t_aligned - t_tgt

        residuals.append(rot_error)
        residuals.append(trans_error)

    return np.concatenate(residuals)


def align_sim3_poses(curr_poses, prev_poses):
    """
    Aligns curr_poses to prev_poses using Sim(3) optimization.

    Returns:
        s: scale (float)
        R_opt: (3, 3) rotation
        t_opt: (3,) translation
    """
    assert curr_poses.shape == prev_poses.shape
    N = curr_poses.shape[0]

    # Initialize: identity rotation, zero translation, unit scale
    init_params = np.zeros(7)
    init_params[6] = 1.0  # scale

    result = least_squares(
        sim3_residual,
        init_params,
        args=(curr_poses, prev_poses),
        method='lm',
        verbose=2
    )

    rotvec = result.x[:3]
    t_opt = result.x[3:6]
    scale = result.x[6]
    R_opt = R.from_rotvec(rotvec).as_matrix()

    return scale, R_opt, t_opt


def transform_points_sim3(points, s, R_opt, t_opt):
    """
    Apply Sim(3) to global 3D points.

    Args:
        points: (N, H, W, 3) or (H, W, 3)
        s: scale (float)
        R_opt: (3, 3) rotation matrix
        t_opt: (3,) translation vector

    Returns:
        Transformed points: same shape
    """
    original_shape = points.shape
    points_flat = points.reshape(-1, 3).T  # (3, H*W*N)
    
    # Apply: x' = s * R * x + t
    points_transformed = s * (R_opt @ points_flat) + t_opt[:, None]  # (3, M)

    return points_transformed.T.reshape(original_shape)  # back to (N, H, W, 3)

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
    # Storage for unified results
    unified_results = {'camera_poses': [], 'images': [], 'conf': [], 'points': []}

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

        predictions['images'] = images[None].permute(0, 1, 3, 4, 2)
        predictions['conf'] = torch.sigmoid(predictions['conf'])
        edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
        predictions['conf'][edge] = 0.0
        del predictions['local_points']

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
        extrinsic_window = predictions['camera_poses']
        points_window = predictions['points']
        # Handle alignment based on window
        if window_idx == 0:
            transformed_extrinsics = extrinsic_window
            transformed_points = points_window
        if window_idx > 0:
            common_frame_idx = start_idx
            print(f"common_frame_idx: {common_frame_idx}")
            prev_overlaps = np.array(unified_results['camera_poses'][common_frame_idx:].copy())
            curr_overlaps = np.array(extrinsic_window[:overlap].copy())

            # s, R, t = calculate_transformation(prev_overlaps, curr_overlaps)
            s, R_opt, t_opt = align_sim3_poses(curr_overlaps, prev_overlaps)
            # Transform all E1 poses
            transformed_extrinsics = transform_poses_sim3(extrinsic_window, s, R_opt, t_opt)[overlap:]
            transformed_points = transform_points_sim3(points_window, s, R_opt, t_opt)[overlap:]
            # debug_alignment(prev_overlaps, curr_overlaps, transformed_extrinsics[:overlap])


   
        for i in range(len(transformed_extrinsics)):
            unified_results['camera_poses'].append(transformed_extrinsics[i])
            unified_results['points'].append(transformed_points[i])
            if window_idx != 0:
                unified_results['images'].append(predictions['images'][overlap + i])
                unified_results['conf'].append(predictions['conf'][overlap + i])
            else:
                unified_results['images'].append(predictions['images'][i])
                unified_results['conf'].append(predictions['conf'][i])

        # Clear GPU memory
        del predictions, images
        
        torch.cuda.empty_cache()
    for key in unified_results.keys():
        unified_results[key] = np.array(unified_results[key])
        print(f"{key}: {unified_results[key].shape}")
    return unified_results

def save_json(predictions, path):

    output = []

    for i in range(len(predictions['camera_poses'])):
        position = predictions['camera_poses'][i][:3, 3].tolist()  # x, y, z
        rot_matrix = predictions['camera_poses'][i][:3, :3]        # 3x3

        # Convert to quaternion: scipy returns (x, y, z, w), reorder to (w, x, y, z)
        quat = R.from_matrix(rot_matrix).as_quat()  # x, y, z, w
        quat = [quat[3], quat[0], quat[1], quat[2]]

        pose_dict = {
            "position": {
                "x": position[0],
                "y": position[1],
                "z": position[2]
            },
            "heading": {
                "w": quat[0],
                "x": quat[1],
                "y": quat[2],
                "z": quat[3]
            }
        }

        output.append(pose_dict)

    with open(path, 'w') as f:
        json.dump(output, f, indent=4)



def main():
    model = load_model()
    os.makedirs(args.save_path, exist_ok=True)
    predictions = run_sliding_window(model, args.data_path, args.window_size, args.overlap)
    
    save_json(predictions, os.path.join(args.save_path, "poses.json"))
    # visualize_camera_poses(predictions['camera_poses'])

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