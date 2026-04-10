import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMICKIT_ROOT = os.path.join(REPO_ROOT, "mimickit")
sys.path.insert(0, MIMICKIT_ROOT)
sys.path.insert(0, REPO_ROOT)

import anim.motion as motion
import anim.motion_lib as motion_lib
import util.torch_util as torch_util


def build_char_model(char_file: str, device: str):
    _, file_ext = os.path.splitext(char_file)
    if file_ext == ".xml":
        import anim.mjcf_char_model as mjcf_char_model

        model = mjcf_char_model.MJCFCharModel(device)
    elif file_ext == ".urdf":
        import anim.urdf_char_model as urdf_char_model

        model = urdf_char_model.URDFCharModel(device)
    elif file_ext == ".usd":
        import anim.usd_char_model as usd_char_model

        model = usd_char_model.USDCharModel(device)
    else:
        raise ValueError(f"Unsupported character file format: {file_ext}")

    model.load(char_file)
    return model


def validate_motion(motion_file: str, char_file: str, device: str) -> None:
    char_model = build_char_model(char_file, device)
    motion_data = motion.load_motion(motion_file)
    frames = np.asarray(motion_data.frames, dtype=np.float32)

    if frames.ndim != 2:
        raise ValueError(f"Expected frames to be rank-2, got shape {frames.shape}")

    expected_width = 6 + char_model.get_dof_size()
    if frames.shape[1] != expected_width:
        raise ValueError(
            f"Frame width mismatch: got {frames.shape[1]}, expected 6 + dof_size = {expected_width} "
            f"for {char_file}"
        )

    root_pos, root_rot_exp, dof_pos = motion_lib.extract_pose_data(frames)
    root_pos_t = torch.tensor(root_pos, dtype=torch.float32, device=device)
    root_rot_t = torch_util.exp_map_to_quat(torch.tensor(root_rot_exp, dtype=torch.float32, device=device))
    dof_pos_t = torch.tensor(dof_pos, dtype=torch.float32, device=device)

    joint_rot = char_model.dof_to_rot(dof_pos_t)
    dof_roundtrip = char_model.rot_to_dof(joint_rot)
    roundtrip_err = torch.max(torch.abs(dof_roundtrip - dof_pos_t)).item()

    body_pos, _ = char_model.forward_kinematics(root_pos_t, root_rot_t, joint_rot)
    if not torch.isfinite(body_pos).all():
        raise ValueError("Forward kinematics produced non-finite body positions.")

    min_body_height = torch.min(body_pos[..., 2]).item()
    max_body_height = torch.max(body_pos[..., 2]).item()
    root_height_min = torch.min(root_pos_t[..., 2]).item()
    root_height_max = torch.max(root_pos_t[..., 2]).item()

    world_up = torch.zeros((root_rot_t.shape[0], 3), dtype=root_rot_t.dtype, device=root_rot_t.device)
    world_up[:, 2] = 1.0
    rotated_up = torch_util.quat_rotate(root_rot_t, world_up)
    upside_down_frac = torch.mean((rotated_up[:, 2] < 0.0).float()).item()

    print("\n" + "=" * 60)
    print("MIMICKIT MOTION VALIDATION")
    print("=" * 60)
    print(f"Motion file:          {motion_file}")
    print(f"Character file:       {char_file}")
    print(f"Frames:               {frames.shape[0]}")
    print(f"Frame width:          {frames.shape[1]}")
    print(f"Expected frame width: {expected_width}")
    print(f"FPS:                  {motion_data.fps}")
    print(f"Loop mode:            {motion_data.loop_mode.name}")
    print(f"DoF roundtrip maxerr: {roundtrip_err:.6e}")
    print(f"Root z range:         [{root_height_min:.4f}, {root_height_max:.4f}]")
    print(f"Body z range:         [{min_body_height:.4f}, {max_body_height:.4f}]")
    print(f"Upside-down fraction: {upside_down_frac:.3f}")
    print("=" * 60)

    if upside_down_frac > 0.25:
        print("WARNING: many frames appear upside-down relative to +Z world up.")
    if min_body_height < -0.25:
        print("WARNING: body positions go well below ground; basis or root height may need repair.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that a MimicKit motion matches a target rig.")
    parser.add_argument("--motion_file", required=True, help="Path to MimicKit .pkl motion")
    parser.add_argument("--char_file", required=True, help="Target rig asset (.xml/.urdf/.usd)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    validate_motion(
        motion_file=args.motion_file,
        char_file=args.char_file,
        device=args.device,
    )


if __name__ == "__main__":
    main()
