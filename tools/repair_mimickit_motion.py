import argparse
import os
import pickle
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))
sys.path.insert(0, REPO_ROOT)

import anim.mjcf_char_model as mjcf_char_model
import anim.motion_lib as motion_lib


def _load_motion(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_motion(path: str, motion: dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(motion, f)


def _extract_joint_limits(char_file: str) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(char_file).getroot()
    body_root = root.find("worldbody").find("body")
    lower = []
    upper = []

    def walk(node: ET.Element) -> None:
        for child in node.findall("body"):
            for joint in child.findall("joint"):
                joint_range = joint.attrib.get("range")
                if joint_range is None:
                    continue
                lo, hi = [float(x) for x in joint_range.split()]
                lower.append(lo)
                upper.append(hi)
            walk(child)

    walk(body_root)
    return np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32)


def _compute_ground_shift(
    motion_file: str,
    char_file: str,
    body_names: list[str],
    samples: int,
    target_height: float,
) -> float:
    device = "cpu"
    char_model = mjcf_char_model.MJCFCharModel(device)
    char_model.load(char_file)
    motion = motion_lib.MotionLib(motion_file, char_model, device)
    name_to_id = {name: i for i, name in enumerate(char_model.get_body_names())}
    body_ids = [name_to_id[name] for name in body_names]

    motion_len = motion.get_motion_lengths()[0].item()
    times = torch.linspace(0.0, motion_len, steps=max(samples, 2))
    min_height = float("inf")
    for t in times:
        root_pos, root_rot, _, _, joint_rot, _ = motion.calc_motion_frame(
            torch.zeros(1, dtype=torch.long),
            torch.tensor([t.item()], dtype=torch.float32),
        )
        body_pos, _ = char_model.forward_kinematics(root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot)
        curr_height = body_pos[0, body_ids, 2].min().item()
        min_height = min(min_height, curr_height)

    return min_height - target_height


def repair_motion(
    input_file: str,
    output_file: str,
    char_file: str,
    clamp_dofs: bool,
    ground_bodies: list[str],
    ground_samples: int,
    ground_target_height: float,
) -> dict:
    motion = _load_motion(input_file)
    frames = np.array(motion["frames"], dtype=np.float32)

    if clamp_dofs:
        dof_lo, dof_hi = _extract_joint_limits(char_file)
        frames[:, 6:] = np.clip(frames[:, 6:], dof_lo, dof_hi)

    if len(ground_bodies) > 0:
        temp_motion = {
            "loop_mode": motion["loop_mode"],
            "fps": motion["fps"],
            "frames": frames.tolist(),
        }
        tmp_path = os.path.join("/tmp", os.path.basename(output_file))
        _save_motion(tmp_path, temp_motion)
        shift = _compute_ground_shift(
            motion_file=tmp_path,
            char_file=char_file,
            body_names=ground_bodies,
            samples=ground_samples,
            target_height=ground_target_height,
        )
        frames[:, 2] -= shift

    repaired = {
        "loop_mode": motion["loop_mode"],
        "fps": motion["fps"],
        "frames": frames.tolist(),
    }
    _save_motion(output_file, repaired)
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair a MimicKit motion by clamping DOFs and grounding the feet.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--char_file", default="data/assets/smpl/smpl.xml")
    parser.add_argument("--no_clamp_dofs", action="store_true")
    parser.add_argument(
        "--ground_bodies",
        nargs="*",
        default=["L_Toe", "R_Toe"],
        help="Bodies used to estimate ground clearance",
    )
    parser.add_argument("--ground_samples", type=int, default=240)
    parser.add_argument("--ground_target_height", type=float, default=0.003)
    args = parser.parse_args()

    repair_motion(
        input_file=args.input_file,
        output_file=args.output_file,
        char_file=args.char_file,
        clamp_dofs=not args.no_clamp_dofs,
        ground_bodies=args.ground_bodies,
        ground_samples=args.ground_samples,
        ground_target_height=args.ground_target_height,
    )


if __name__ == "__main__":
    main()
