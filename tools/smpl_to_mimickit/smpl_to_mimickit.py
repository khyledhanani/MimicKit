"""
This module provides functionality to convert motion data from SMPL format to MimicKit format for the SMPL humanoid. 

Usage:
    Command line:
        python tools/data_format/smpl_to_mimickit.py
    Required arguments:
        --input_file PATH       Path to the input SMPL npz file
        --output_file PATH      Path to save the output MimicKit pickle file
    Optional arguments:
        --loop {wrap,clamp}     Loop mode for the motion (default: wrap)
        --start_frame INT       Start frame for motion clipping (default: 0)
        --end_frame INT         End frame for motion clipping (default: -1, uses all frames)
        --output_fps INT        Frame rate for the output motion (default: same as input)
    
SMPL Format:
    The input SMPL format should be a npz file containing either:
    - AMASS-style arrays with keys:
      - 'poses': Pose parameters array, shape (num_frames, num_pose_params)
      - 'trans': Translation array, shape (num_frames, 3)
      - 'mocap_framerate' or 'fps': Frame rate (int)
    - Or interaction-export arrays with keys:
      - 'global_orient' or 'root_orient': Root orientation, shape (num_frames, 3)
      - 'pose_body': Body pose, shape (num_frames, 21, 3) or (num_frames, 63)
      - 'trans': Translation array, shape (num_frames, 3)
      - optional 'fps' or a CLI-provided input fps

Output:
    Creates a dictionary containing MimicKit motion data saved as a pickle file, with loop mode stored as INT and motion data stored as
    concatenated arrays of [root_pos, root_rot_expmap, dof_pos] per frame.
"""

import argparse
import numpy as np
import sys
import torch
import os

sys.path.append(".")

from mimickit.anim.motion import Motion, LoopMode
from mimickit.util.torch_util import (
    quat_to_exp_map,
    exp_map_to_quat,
    quat_mul,
    quat_conjugate,
)
from tools.smpl_to_mimickit.smpl_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from tools.smpl_to_mimickit.smpl_constants import PARENT_INDICES, LOCAL_TRANSLATION
from tools.smpl_to_mimickit.rotation_tools import compute_global_rotations, compute_local_rotations, compute_global_translations

# Two coordinate-frame quaternions are needed:
#
# 1) YUP_TO_ZUP / YUP_TO_ZUP_INV  (F = 90° around +X, maps Y→Z and Z→-Y)
#    Used for:
#      • Transforming the root rotation via conjugation F*R*F^{-1}, which
#        re-expresses the global pelvis orientation in Z-up while preserving
#        the physical direction.
#      • Computing world-space FK joint positions for the z_correction step.
#    R_Z_NEG90 is applied on top of the conjugated root to shift the SMPL
#    T-pose rest facing direction from -Y to +X (MuJoCo SMPL default).
#
# 2) YUP_TO_ZUP_DOF  (= 120° around (-1,-1,-1)/√3, cyclic permutation X→Z→Y→X)
#    Used for: converting body-joint DOF rotations (parent-relative) from
#    SMPL Y-up axes to MuJoCo Z-up axes.
#    In SMPL Y-up (body facing -Z, up +Y) the knee-flex axis is local-X.
#    In MuJoCo Z-up (body facing +X, up +Z) the knee-flex axis is local-Y.
#    The 120° cyclic permutation (X→Y, Y→Z, Z→X applied to rotation axes via
#    conjugation ZUP_TO_YUP·R·YUP_TO_ZUP_DOF) achieves exactly this mapping.
#    Equivalently: post-multiplying every global rotation by YUP_TO_ZUP_DOF
#    before extracting parent-relative locals gives the same result.
YUP_TO_ZUP     = torch.tensor([0.7071068,  0.0, 0.0, 0.7071068])   # (x,y,z,w), 90° around +X
YUP_TO_ZUP_INV = torch.tensor([-0.7071068, 0.0, 0.0, 0.7071068])   # conjugate, -90° around +X
R_Z_NEG90      = torch.tensor([0.0, 0.0, -0.7071068, 0.7071068])   # -90° around +Z
YUP_TO_ZUP_DOF = torch.tensor([-0.5, -0.5, -0.5, 0.5])             # 120° around (-1,-1,-1)/√3

# Used for "zup" coord_system mode (mpc_joints format).
# In that format root_orient is stored as q_mujoco * inv(INTERX_ROOT_BASIS_CORRECTION),
# so post-multiplying by INTERX_ROOT_BASIS_CORRECTION recovers the MuJoCo root rotation.
INTERX_ROOT_BASIS_CORRECTION = torch.tensor([-0.5, -0.5, -0.5, 0.5])  # 120° around (-1,-1,-1)/√3


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
        poses: numpy array of shape (N, num_pose_params)
        trans: numpy array of shape (N, 3)
        fps: frame rate
    """
    if input_file.endswith(".npz"):
        data = np.load(input_file, allow_pickle=True)
        trans = data["trans"]  # (N, 3)

        if "poses" in data:
            poses = data["poses"]  # (N, num_pose_params)
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
    else:
        raise ValueError("Unsupported file format. Please provide a .npz file.")
    
    return poses, trans, fps


def convert_smpl_to_mimickit(input_file: str, 
                             output_file: str, 
                             loop_mode: str = "clamp", 
                             start_frame: int = 0, 
                             end_frame: int = -1, 
                             output_fps: int = -1,
                             z_correction: str = "none",
                             input_fps: int = -1,
                             coord_system: str = "yup") -> Motion:
    """
    Convert SMPL/AMASS motion data to MimicKit format.
    
    Args:
        input_file: Path to input SMPL motion file (.npz)
        output_file: Path to output MimicKit pickle file
        loop_mode: "wrap" or "clamp"
        start_frame: Start frame for clipping
        end_frame: End frame for clipping (-1 for all)
        output_fps: Output frame rate (-1 to use source fps)
        z_correction: Z-axis correction method ("none", "calibrate", "full")
        coord_system: Input coordinate system convention:
            - "yup": Standard SMPL / InterX / AMASS format.  trans is in Y-up world
              space (Y = height ~1.24 m); root_orient is expressed in the SMPL Y-up
              basis.  The converter applies X-90° conjugation + R_Z_NEG90 to produce
              the MuJoCo root rotation.
            - "zup": MPC-joints format.  trans is *already* in Z-up world space
              (Z = height); root_orient is stored in a pre-rotated basis where
              post-multiplying by INTERX_ROOT_BASIS_CORRECTION gives the MuJoCo root
              rotation.  Body joint DOFs (pose_body) remain in SMPL Y-up axes and
              still receive the 120° cyclic transformation.

    Returns:
        MimicKit Motion object

    Notes:
        There are three options for z_correction:
        - "none": No correction applied.
        - "calibrate": Adjusts the vertical position based on the lowest foot position in the first 30 frames.
        - "full": Adjusts the vertical position based on the lowest foot position across the entire motion.
    """
    # Parse loop mode
    if loop_mode == "wrap":
        loop_mode_out = LoopMode.WRAP
    elif loop_mode == "clamp":
        loop_mode_out = LoopMode.CLAMP
    else:
        raise ValueError(f"Invalid loop_mode: {loop_mode}. Choose 'wrap' or 'clamp'.")
    
    # Load SMPL data
    poses, trans, fps = load_smpl_motion(input_file, input_fps=input_fps)
    N = poses.shape[0]
    
    print("\n" + "="*60)
    print("📥 LOADED SMPL DATA")
    print("="*60)
    print(f"📁 File: {input_file}")
    print(f"🎬 Frames: {N}")
    print(f"⏱️  FPS: {fps}")
    print(f"🦴 Pose params: {poses.shape[1]}")
    print(f"📍 Trans shape: {trans.shape}")
    print("="*60 + "\n")

    root_rot = exp_map_to_quat(torch.tensor(poses[:, 0:3], dtype=torch.float32)).numpy()

    if coord_system == "yup":
        # Convert trans from SMPL Y-up to MimicKit Z-up: X unchanged, new_Y = -old_Z, new_Z = old_Y
        # InterX trans is in Y-up space (Y = height ~1.24m); MimicKit expects Z-up (Z = height).
        new_trans = trans.copy()
        new_trans[:, 1] = -trans[:, 2]  # new_Y = -old_Z (depth becomes forward, negated)
        new_trans[:, 2] = trans[:, 1]   # new_Z = old_Y  (height maps to Z)
        trans = new_trans
    # coord_system == "zup": trans is already in Z-up world space — use it as-is.

    pose_aa = np.concatenate([poses[:, :66], np.zeros((trans.shape[0], 6))], axis = -1) # Keep only SMPL parameters without hands, and explicitly set hand dofs to zero

    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
    pose_quat = exp_map_to_quat(torch.tensor(pose_aa_mj.reshape(-1, 3), dtype=torch.float32)).numpy().reshape(N, 24, 4)

    global_rot = compute_global_rotations(
        torch.tensor(pose_quat, dtype=torch.float32),
        PARENT_INDICES
    )
    n_flat = global_rot.reshape(-1, 4).shape[0]

    # --- Body-joint DOFs ---
    # Post-multiply each global rotation by YUP_TO_ZUP_DOF (120° cyclic), then
    # extract parent-relative locals.  This conjugates each local rotation by
    # ZUP_TO_YUP (the inverse), which cyclically permutes the rotation axes
    # X→Y, Y→Z, Z→X — mapping SMPL Y-up joint axes to MuJoCo Z-up joint axes.
    #
    # For "zup" (mpc_joints) format: global_rot[0] = q_mpc_root * YUP_TO_ZUP_DOF
    #   = q_mpc_root * INTERX_ROOT_BASIS_CORRECTION = q_mujoco.
    # So rotated_local_rot[0] already holds the correct MuJoCo root quaternion.
    rotated_global_rot_dof = quat_mul(
        global_rot.reshape(-1, 4),
        YUP_TO_ZUP_DOF.expand(n_flat, -1)
    ).reshape(N, -1, 4)
    rotated_local_rot = compute_local_rotations(
        rotated_global_rot_dof,
        PARENT_INDICES
    )

    # --- FK positions (used only for z_correction) ---
    if coord_system == "yup":
        # Conjugation F*R*F^{-1} re-expresses global orientations in Z-up so that
        # FK positions computed with LOCAL_TRANSLATION (Z-up) give world Z heights.
        rotated_global_rot_for_fk = quat_mul(
            quat_mul(YUP_TO_ZUP.expand(n_flat, -1), global_rot.reshape(-1, 4)),
            YUP_TO_ZUP_INV.expand(n_flat, -1)
        ).reshape(N, -1, 4)
    else:
        # "zup" (mpc_joints): rotated_local_rot[0] = q_mujoco (correct Z-up root).
        # Recomputing global rotations from these Z-up local rotations yields the
        # correct world-space orientations for FK height computation.
        rotated_global_rot_for_fk = compute_global_rotations(rotated_local_rot, PARENT_INDICES)

    global_translation = compute_global_translations(
        rotated_global_rot_for_fk,
        torch.tensor(LOCAL_TRANSLATION, dtype=torch.float32),
        PARENT_INDICES
    ).numpy()

    global_translation += trans[:, None, :]

    dof_pos = quat_to_exp_map(rotated_local_rot[:, 1:, :]).numpy().reshape(N, -1)

    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)
    N_r = root_rot_t.shape[0]
    if coord_system == "yup":
        # Conjugate to re-express in Z-up, then apply rest-pose correction (-90° around Z)
        # so that the MuJoCo T-pose facing (+X) matches the converted SMPL T-pose.
        rotated_root_rot_quat = quat_mul(
            R_Z_NEG90.expand(N_r, -1),
            quat_mul(
                quat_mul(YUP_TO_ZUP.expand(N_r, -1), root_rot_t),
                YUP_TO_ZUP_INV.expand(N_r, -1)
            )
        )
    else:
        # "zup" / mpc_joints format: root_orient is stored as q_mujoco * inv(correction),
        # so post-multiplying by INTERX_ROOT_BASIS_CORRECTION recovers the MuJoCo root.
        rotated_root_rot_quat = quat_mul(
            root_rot_t,
            INTERX_ROOT_BASIS_CORRECTION.expand(N_r, -1)
        )
    root_rot = quat_to_exp_map(rotated_root_rot_quat).numpy()

    # Z-correction
    if z_correction == "full":
        min_height = np.min(global_translation[:, :, 2])
        trans[:, 2] -= min_height - 0.025   # Adjust for foot mesh height
    elif z_correction == "calibrate":
        min_height = np.min(global_translation[:30, :, 2])
        trans[:, 2] -= min_height - 0.025   # Adjust for foot mesh height

    frames = np.concatenate([trans, root_rot, dof_pos], axis=1)
    
    # Clip frames
    if end_frame == -1:
        end_frame = frames.shape[0]
    assert 0 <= start_frame < end_frame <= frames.shape[0], \
        f"Invalid frame range: [{start_frame}, {end_frame}] for {frames.shape[0]} frames"
    frames = frames[start_frame:end_frame, :]
    
    # Set output fps
    save_fps = fps if output_fps == -1 else output_fps
    
    motion = Motion(loop_mode=loop_mode_out, fps=save_fps, frames=frames)

    # Check if directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    motion.save(output_file)
    
    print("\n" + "="*60)
    print("✅ CONVERSION SUCCESSFUL")
    print("="*60)
    print(f"📁 Input:  {input_file}")
    print(f"💾 Output: {output_file}")
    print("-"*60)
    print(f"📊 Frames shape: {frames.shape}")
    print(f"🎬 Total frames: {frames.shape[0]}")
    print(f"⏱️  FPS: {save_fps}")
    print(f"🔄 Loop mode: {loop_mode_out}")
    print(f"🦴 DoFs: {69}")
    print("="*60 + "\n")
    
    return motion


def main():
    parser = argparse.ArgumentParser(description="Convert SMPL/AMASS motion data to MimicKit format for the SMPL humanoid.")
    parser.add_argument("--input_file", required=True,help="Path to input SMPL motion file (.npz or .npy)")
    parser.add_argument("--output_file", required=True, help="Path to output MimicKit pickle file")
    parser.add_argument("--loop", default="wrap", choices=["wrap", "clamp"], help="Loop mode for the motion (default: wrap)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for clipping (default: 0)")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame for clipping (default: -1, uses all frames)")
    parser.add_argument("--output_fps", type=int, default=-1, help="Output frame rate (default: -1, uses source fps)")
    parser.add_argument("--input_fps", type=int, default=-1, help="Input frame rate override when not stored in the npz")
    parser.add_argument("--z_correction", type=str, default="calibrate", choices=["none", "calibrate", "full"], help="Z-axis correction method (default: calibrate)")
    parser.add_argument("--coord_system", type=str, default="yup", choices=["yup", "zup"],
                        help="Input coordinate system: 'yup' for standard SMPL/InterX/AMASS (default), "
                             "'zup' for mpc_joints format where trans is already in Z-up and "
                             "root_orient uses the INTERX_ROOT_BASIS_CORRECTION convention.")
    
    args = parser.parse_args()
    
    convert_smpl_to_mimickit(
        args.input_file, 
        args.output_file,
        loop_mode=args.loop,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        output_fps=args.output_fps,
        z_correction=args.z_correction,
        input_fps=args.input_fps,
        coord_system=args.coord_system,
    )


if __name__ == "__main__":
    main()
