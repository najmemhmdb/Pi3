import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import os
from demo_gradio import predictions_to_glb

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='/mnt/public/Ehsan/datasets/private/Najmeh/real_data/kashiwa/output_processed/camera',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
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
    print(imgs.shape)
    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(imgs[None]) # Add batch dimension


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

    glbfile = os.path.join(
        "./",
        f"glbscene.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=3.0,
        filter_by_frames="All",
        show_cam=True,
    )
    glbscene.export(file_obj=glbfile)

    # Clean up
    # # 4. process mask
    # masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    # non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    # masks = torch.logical_and(masks, non_edge)[0]

    # # 5. Save points
    # print(f"Saving point cloud to: {args.save_path}")
    # write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], args.save_path)
    # print("Done.")