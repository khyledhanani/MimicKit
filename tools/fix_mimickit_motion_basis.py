import argparse
import math
import os
import pickle
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from mimickit.util.torch_util import exp_map_to_quat, quat_mul, quat_rotate, quat_to_exp_map


X_POS_90 = torch.tensor([math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)], dtype=torch.float32)
X_180 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def fix_motion_basis(input_file: str, output_file: str) -> None:
    with open(input_file, "rb") as f:
        motion = pickle.load(f)

    frames = np.array(motion["frames"], dtype=np.float32)
    root_pos = torch.tensor(frames[:, :3], dtype=torch.float32)
    root_rot = exp_map_to_quat(torch.tensor(frames[:, 3:6], dtype=torch.float32))

    correction = X_POS_90.expand(frames.shape[0], -1)
    fixed_root_pos = quat_rotate(correction, root_pos)
    fixed_root_rot = quat_mul(root_rot, correction)

    world_up = torch.zeros((frames.shape[0], 3), dtype=torch.float32)
    world_up[:, 2] = 1.0
    rotated_up = quat_rotate(fixed_root_rot, world_up)
    upside_down = rotated_up[:, 2] < 0.0
    if torch.any(upside_down):
        fixed_root_rot = fixed_root_rot.clone()
        fixed_root_rot[upside_down] = quat_mul(
            fixed_root_rot[upside_down],
            X_180.expand(int(upside_down.sum().item()), -1),
        )

    fixed_root_exp = quat_to_exp_map(fixed_root_rot)

    fixed_frames = frames.copy()
    fixed_frames[:, :3] = fixed_root_pos.cpu().numpy()
    fixed_frames[:, 3:6] = fixed_root_exp.cpu().numpy()

    out_motion = {
        "loop_mode": motion["loop_mode"],
        "fps": motion["fps"],
        "frames": fixed_frames.tolist(),
    }

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(out_motion, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix MimicKit motion files exported in a Y-up root basis.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    fix_motion_basis(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
