import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import os
from demo_gradio import predictions_to_glb
import json
import imageio
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R


def save_camera_poses_as_quaternion(predictions, output_path):
    poses = predictions['camera_poses']  # (N, 4, 4)
    camera_list = []

    for pose in poses:
        # Extract translation
        t = pose[:3, 3]
        position = {
            "x": float(t[0]),
            "y": float(t[1]),
            "z": float(t[2])
        }

        # Extract rotation matrix and convert to quaternion
        R_mat = pose[:3, :3]
        quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
        heading = {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3])
        }

        camera_list.append({
            "position": position,
            "heading": heading
        })

    with open(output_path, 'w') as f:
        json.dump(camera_list, f, indent=4)


def save_depth_maps(predictions, output_dir, width=1920, height=1080):
    os.makedirs(output_dir, exist_ok=True)
    points = predictions['points']  # (N, H, W, 3)
    poses = predictions['camera_poses']  # (N, 4, 4)

    fx, fy = predictions['fl_x'], predictions['fl_y']
    cx, cy = predictions['cx'], predictions['cy']

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    for i in tqdm(range(len(points))):
        # Points: (H, W, 3)
        pts = points[i].reshape(-1, 3)
        pose = poses[i]  # (4, 4)

        # Transform to camera frame
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)
        cam_pts = (np.linalg.inv(pose) @ pts_h.T).T[:, :3]

        # Filter positive z (visible)
        valid = cam_pts[:, 2] > 0
        cam_pts = cam_pts[valid]

        # Project to 2D
        pix = (K @ cam_pts.T).T
        pix[:, 0] /= pix[:, 2]
        pix[:, 1] /= pix[:, 2]
        pix = pix[:, :2].astype(np.int32)
        depth = cam_pts[:, 2]

        # Create empty depth map
        depth_map = np.zeros((height, width), dtype=np.float32)

        # Fill depth values (use z-buffering)
        for (u, v), d in zip(pix, depth):
            if 0 <= u < width and 0 <= v < height:
                if depth_map[v, u] == 0 or d < depth_map[v, u]:
                    depth_map[v, u] = d

        # Save depth map
        depth_path = os.path.join(output_dir, f"{i:03d}.png")
        imageio.imwrite(depth_path, (depth_map * 1000).astype(np.uint16))  # scale to mm

def save_json(predictions, path):
    predictions['fl_x'], predictions['fl_y'] = 1236.78, 1201.39
    predictions['cx'], predictions['cy'] = 971.40, 565.10
    predictions['k1'], predictions['k2'], predictions['p1'], predictions['p2'], predictions['k3'] = 0.0220, -0.0323, 0.0055, 0.0037, 0.0139
    output = {"frames": []}

    for i in range(len(predictions['points'])):
        frame = {}
        frame['file_path'] = f"camera/{i:03}.jpg"
        frame['transform_matrix'] = predictions['camera_poses'][i].tolist()
        frame['w'] = 1920
        frame['h'] = 1080
        frame['fl_x'] = predictions['fl_x']
        frame['fl_y'] = predictions['fl_y']
        frame['cx'] = predictions['cx']
        frame['cy'] = predictions['cy']
        frame['k1'] = predictions['k1']
        frame['k2'] = predictions['k2']
        frame['p1'] = predictions['p1']
        frame['p2'] = predictions['p2']
        frame['k3'] = predictions['k3']
        frame['camera_model'] = "OPENCV"
        output['frames'].append(frame)


    with open(path, 'w') as f:
        json.dump(output, f, indent=4)
    return predictions

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='/mnt/public/Ehsan/datasets/private/Najmeh/simulated_data/simulated_pandaset_V5_large/001/camera/front_camera',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='./outputs/simulated_pandaset_V5_large/001/front_camera',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path

    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)
    imgs = imgs[:10]
    # 3. create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # 4. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(imgs[None]) # Add batch dimensionx


    predictions['images'] = imgs[None].permute(0, 1, 3, 4, 2)
    predictions['conf'] = torch.sigmoid(predictions['conf'])
    edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
    predictions['conf'][edge] = 0.0
    del predictions['local_points']

    # # transform to first camera coordinate
    # predictions['points'] = torch.einsum('bij, bnhwj -> bnhwi', se3_inverse(predictions['camera_poses'][:, 0]), homogenize_points(predictions['points']))[..., :3]
    # predictions['camera_poses'] = torch.einsum('bij, bnjk -> bnik', se3_inverse(predictions['camera_poses'][:, 0]), predictions['camera_poses'])

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    print(predictions['camera_poses'][:3])
    print(predictions['points'].shape)
    print(predictions['conf'].shape)
    exit(0)
    save_camera_poses_as_quaternion(predictions, os.path.join(args.save_path, "poses.json"))

    glbfile = os.path.join(
        args.save_path,
        f"glbscene.glb",
    )

    # save predicitons as transforms.json 
    # predictions = save_json(predictions, "/mnt/public/Ehsan/datasets/private/Najmeh/real_data/kashiwa/output_processed/transforms.json")
    # save_depth_maps(predictions, output_dir="/mnt/public/Ehsan/datasets/private/Najmeh/real_data/kashiwa/output_processed/depths")
    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=3.0,
        filter_by_frames="All",
        show_cam=True,
    )
    glbscene.export(file_obj=glbfile)
