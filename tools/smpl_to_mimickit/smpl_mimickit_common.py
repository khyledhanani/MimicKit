"""
Shared implementation for SMPL-parameterized → MimicKit motion conversion.

Two public entry points:
  - convert_interx_smpl_to_mimickit: Y-up SMPL / InterX / AMASS exports
  - convert_mpc_joints_to_mimickit: mpc_joints exports (Z-up trans, basis-corrected root)
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch

from mimickit.anim.motion import Motion, LoopMode
from mimickit.util.torch_util import (
    quat_to_exp_map,
    exp_map_to_quat,
    quat_mul,
)
from tools.smpl_to_mimickit.smpl_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from tools.smpl_to_mimickit.smpl_constants import PARENT_INDICES, LOCAL_TRANSLATION
from tools.smpl_to_mimickit.rotation_tools import (
    compute_global_rotations,
    compute_local_rotations,
    compute_global_translations,
)

# 90° around +X: Y→Z, Z→-Y (re-express SMPL global orientations for Z-up FK)
YUP_TO_ZUP = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
YUP_TO_ZUP_INV = torch.tensor([-0.7071068, 0.0, 0.0, 0.7071068])
R_Z_NEG90 = torch.tensor([0.0, 0.0, -0.7071068, 0.7071068])
# 120° around (-1,-1,-1)/√3: cyclic joint-axis mapping SMPL Y-up → MuJoCo Z-up DOFs
YUP_TO_ZUP_DOF = torch.tensor([-0.5, -0.5, -0.5, 0.5])
# Same quaternion as YUP_TO_ZUP_DOF; mpc root_orient is q_mujoco * inv(this)
INTERX_ROOT_BASIS_CORRECTION = YUP_TO_ZUP_DOF


def _parse_fps(data, input_fps: int) -> int:
    fps = data.get("mocap_framerate", data.get("fps", input_fps))
    if fps is None or fps == -1:
        raise ValueError(
            "Could not determine input fps from file metadata. "
            "Provide --input_fps."
        )
    if hasattr(fps, "item"):
        fps = fps.item()
    return int(fps)


def load_smpl_motion(input_file: str, input_fps: int = -1) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load SMPL/AMASS motion data from .npz file.

    Returns:
        poses: (N, num_pose_params)
        trans: (N, 3)
        fps: int
    """
    if not input_file.endswith(".npz"):
        raise ValueError("Unsupported file format. Please provide a .npz file.")

    data = np.load(input_file, allow_pickle=True)
    trans = data["trans"]

    if "poses" in data:
        poses = data["poses"]
    elif ("global_orient" in data or "root_orient" in data) and "pose_body" in data:
        root_orient = data["global_orient"] if "global_orient" in data else data["root_orient"]
        pose_body = data["pose_body"]
        if pose_body.ndim == 3:
            pose_body = pose_body.reshape(pose_body.shape[0], -1)
        poses = np.concatenate([root_orient, pose_body], axis=-1)
    else:
        raise KeyError(
            "Unsupported npz schema. Expected either 'poses' or "
            "'global_orient'/'root_orient' + 'pose_body', along with 'trans'."
        )

    fps = _parse_fps(data, input_fps)
    return poses, trans, fps


def _parse_loop_mode(loop_mode: str) -> LoopMode:
    if loop_mode == "wrap":
        return LoopMode.WRAP
    if loop_mode == "clamp":
        return LoopMode.CLAMP
    raise ValueError(f"Invalid loop_mode: {loop_mode}. Choose 'wrap' or 'clamp'.")


def _convert_smpl_npz_to_mimickit(
    input_file: str,
    output_file: str,
    *,
    format_kind: Literal["interx_smpl", "mpc_joints"],
    loop_mode: str = "clamp",
    start_frame: int = 0,
    end_frame: int = -1,
    output_fps: int = -1,
    z_correction: str = "none",
    input_fps: int = -1,
) -> Motion:
    loop_mode_out = _parse_loop_mode(loop_mode)
    poses, trans, fps = load_smpl_motion(input_file, input_fps=input_fps)
    N = poses.shape[0]

    print("\n" + "=" * 60)
    print("📥 LOADED SMPL DATA")
    print("=" * 60)
    print(f"📁 File: {input_file}")
    print(f"🎬 Format: {format_kind}")
    print(f"🎬 Frames: {N}")
    print(f"⏱️  FPS: {fps}")
    print(f"🦴 Pose params: {poses.shape[1]}")
    print(f"📍 Trans shape: {trans.shape}")
    print("=" * 60 + "\n")

    root_rot = exp_map_to_quat(torch.tensor(poses[:, 0:3], dtype=torch.float32)).numpy()

    if format_kind == "interx_smpl":
        new_trans = trans.copy()
        new_trans[:, 1] = -trans[:, 2]
        new_trans[:, 2] = trans[:, 1]
        trans = new_trans
    # mpc_joints: trans already Z-up

    pose_aa = np.concatenate(
        [poses[:, :66], np.zeros((trans.shape[0], 6))],
        axis=-1,
    )

    smpl_2_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES
    ]
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
    pose_quat = (
        exp_map_to_quat(torch.tensor(pose_aa_mj.reshape(-1, 3), dtype=torch.float32))
        .numpy()
        .reshape(N, 24, 4)
    )

    global_rot = compute_global_rotations(
        torch.tensor(pose_quat, dtype=torch.float32),
        PARENT_INDICES,
    )
    n_flat = global_rot.reshape(-1, 4).shape[0]

    rotated_global_rot_dof = quat_mul(
        global_rot.reshape(-1, 4),
        YUP_TO_ZUP_DOF.expand(n_flat, -1),
    ).reshape(N, -1, 4)
    rotated_local_rot = compute_local_rotations(rotated_global_rot_dof, PARENT_INDICES)

    if format_kind == "interx_smpl":
        rotated_global_rot_for_fk = quat_mul(
            quat_mul(YUP_TO_ZUP.expand(n_flat, -1), global_rot.reshape(-1, 4)),
            YUP_TO_ZUP_INV.expand(n_flat, -1),
        ).reshape(N, -1, 4)
    else:
        rotated_global_rot_for_fk = compute_global_rotations(rotated_local_rot, PARENT_INDICES)

    global_translation = compute_global_translations(
        rotated_global_rot_for_fk,
        torch.tensor(LOCAL_TRANSLATION, dtype=torch.float32),
        PARENT_INDICES,
    ).numpy()
    global_translation += trans[:, None, :]

    dof_pos = quat_to_exp_map(rotated_local_rot[:, 1:, :]).numpy().reshape(N, -1)

    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)
    N_r = root_rot_t.shape[0]
    if format_kind == "interx_smpl":
        rotated_root_rot_quat = quat_mul(
            R_Z_NEG90.expand(N_r, -1),
            quat_mul(
                quat_mul(YUP_TO_ZUP.expand(N_r, -1), root_rot_t),
                YUP_TO_ZUP_INV.expand(N_r, -1),
            ),
        )
    else:
        rotated_root_rot_quat = quat_mul(
            root_rot_t,
            INTERX_ROOT_BASIS_CORRECTION.expand(N_r, -1),
        )
    root_rot = quat_to_exp_map(rotated_root_rot_quat).numpy()

    if z_correction == "full":
        min_height = np.min(global_translation[:, :, 2])
        trans[:, 2] -= min_height - 0.025
    elif z_correction == "calibrate":
        min_height = np.min(global_translation[:30, :, 2])
        trans[:, 2] -= min_height - 0.025

    frames = np.concatenate([trans, root_rot, dof_pos], axis=1)

    if end_frame == -1:
        end_frame = frames.shape[0]
    assert 0 <= start_frame < end_frame <= frames.shape[0], (
        f"Invalid frame range: [{start_frame}, {end_frame}] for {frames.shape[0]} frames"
    )
    frames = frames[start_frame:end_frame, :]

    save_fps = fps if output_fps == -1 else output_fps
    motion = Motion(loop_mode=loop_mode_out, fps=save_fps, frames=frames)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    motion.save(output_file)

    print("\n" + "=" * 60)
    print("✅ CONVERSION SUCCESSFUL")
    print("=" * 60)
    print(f"📁 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print("-" * 60)
    print(f"📊 Frames shape: {frames.shape}")
    print(f"🎬 Total frames: {frames.shape[0]}")
    print(f"⏱️  FPS: {save_fps}")
    print(f"🔄 Loop mode: {loop_mode_out}")
    print(f"🦴 DoFs: {69}")
    print("=" * 60 + "\n")

    return motion


def convert_interx_smpl_to_mimickit(
    input_file: str,
    output_file: str,
    loop_mode: str = "clamp",
    start_frame: int = 0,
    end_frame: int = -1,
    output_fps: int = -1,
    z_correction: str = "none",
    input_fps: int = -1,
) -> Motion:
    """
    InterX / AMASS / standard SMPL Y-up exports: trans has Y = height; root_orient in SMPL Y-up.
    """
    return _convert_smpl_npz_to_mimickit(
        input_file,
        output_file,
        format_kind="interx_smpl",
        loop_mode=loop_mode,
        start_frame=start_frame,
        end_frame=end_frame,
        output_fps=output_fps,
        z_correction=z_correction,
        input_fps=input_fps,
    )


def convert_mpc_joints_to_mimickit(
    input_file: str,
    output_file: str,
    loop_mode: str = "clamp",
    start_frame: int = 0,
    end_frame: int = -1,
    output_fps: int = -1,
    z_correction: str = "none",
    input_fps: int = -1,
) -> Motion:
    """
    mpc_joints exports: trans already Z-up (Z = height); root_orient uses INTERX_ROOT_BASIS_CORRECTION.
    """
    return _convert_smpl_npz_to_mimickit(
        input_file,
        output_file,
        format_kind="mpc_joints",
        loop_mode=loop_mode,
        start_frame=start_frame,
        end_frame=end_frame,
        output_fps=output_fps,
        z_correction=z_correction,
        input_fps=input_fps,
    )


def convert_smpl_to_mimickit(
    input_file: str,
    output_file: str,
    loop_mode: str = "clamp",
    start_frame: int = 0,
    end_frame: int = -1,
    output_fps: int = -1,
    z_correction: str = "none",
    input_fps: int = -1,
    coord_system: str = "yup",
) -> Motion:
    """
    Backward-compatible dispatcher. Prefer convert_interx_smpl_to_mimickit /
    convert_mpc_joints_to_mimickit or the dedicated CLI scripts.
    """
    import warnings

    if coord_system == "zup":
        warnings.warn(
            "coord_system='zup' is deprecated; use convert_mpc_joints_to_mimickit() "
            "or tools/smpl_to_mimickit/mpc_joints_to_mimickit.py",
            DeprecationWarning,
            stacklevel=2,
        )
        return convert_mpc_joints_to_mimickit(
            input_file,
            output_file,
            loop_mode=loop_mode,
            start_frame=start_frame,
            end_frame=end_frame,
            output_fps=output_fps,
            z_correction=z_correction,
            input_fps=input_fps,
        )
    return convert_interx_smpl_to_mimickit(
        input_file,
        output_file,
        loop_mode=loop_mode,
        start_frame=start_frame,
        end_frame=end_frame,
        output_fps=output_fps,
        z_correction=z_correction,
        input_fps=input_fps,
    )
