import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

# Ensure MimicKit local modules win over any similarly named site-packages.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMICKIT_ROOT = os.path.join(REPO_ROOT, "mimickit")
sys.path.insert(0, MIMICKIT_ROOT)
sys.path.insert(0, REPO_ROOT)

import anim.mjcf_char_model as mjcf_char_model
import anim.motion as motion
import anim.motion_lib as motion_lib
import engines.engine as engine
import engines.engine_builder as engine_builder


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_char_model(char_file, device):
    model = mjcf_char_model.MJCFCharModel(device)
    model.load(char_file)
    return model


def set_char_pose(sim_engine, char_model, obj_id, root_pos, root_rot, joint_rot, joint_dof):
    sim_engine.set_root_pos(None, obj_id, root_pos)
    sim_engine.set_root_rot(None, obj_id, root_rot)
    sim_engine.set_root_vel(None, obj_id, 0.0)
    sim_engine.set_root_ang_vel(None, obj_id, 0.0)
    sim_engine.set_dof_pos(None, obj_id, joint_dof)
    sim_engine.set_dof_vel(None, obj_id, 0.0)

    body_pos, body_rot = char_model.forward_kinematics(
        root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot
    )
    sim_engine.set_body_pos(None, obj_id, body_pos)
    sim_engine.set_body_rot(None, obj_id, body_rot)
    sim_engine.set_body_vel(None, obj_id, 0.0)
    sim_engine.set_body_ang_vel(None, obj_id, 0.0)


def sample_motion(motion_library, motion_time):
    motion_ids = torch.zeros(1, dtype=torch.long, device=motion_time.device)
    return motion_library.calc_motion_frame(motion_ids, motion_time)


def main():
    parser = argparse.ArgumentParser(description="Visualize two MimicKit motion clips in sync.")
    parser.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    parser.add_argument("--char_file", default="data/assets/smpl/smpl.xml")
    parser.add_argument("--motion_file_a", required=True)
    parser.add_argument("--motion_file_b", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time_scale", type=float, default=1.0)
    parser.add_argument("--repeat", type=str, default="true")
    parser.add_argument("--camera_height", type=float, default=1.2)
    parser.add_argument("--camera_distance", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    repeat = args.repeat.lower() in ("1", "true", "t", "yes", "y")
    device = args.device

    engine_config = load_yaml(args.engine_config)
    engine_config["sim_freq"] = engine_config["control_freq"]
    sim_engine = engine_builder.build_engine(
        engine_config, num_envs=1, device=device, visualize=True
    )

    char_model = build_char_model(args.char_file, device)
    motion_a = motion_lib.MotionLib(args.motion_file_a, char_model, device)
    motion_b = motion_lib.MotionLib(args.motion_file_b, char_model, device)

    env_id = sim_engine.create_env()
    assert env_id == 0
    char_a = sim_engine.create_obj(
        env_id=env_id,
        obj_type=engine.ObjType.articulated,
        asset_file=args.char_file,
        name="character_a",
        is_visual=True,
        enable_self_collisions=False,
        disable_motors=True,
        color=np.array([0.25, 0.55, 0.95]),
    )
    char_b = sim_engine.create_obj(
        env_id=env_id,
        obj_type=engine.ObjType.articulated,
        asset_file=args.char_file,
        name="character_b",
        is_visual=True,
        enable_self_collisions=False,
        disable_motors=True,
        color=np.array([0.35, 0.8, 0.35]),
    )
    sim_engine.initialize_sim()

    motion_len_a = motion_a.get_motion_length(torch.zeros(1, dtype=torch.long, device=device))
    motion_len_b = motion_b.get_motion_length(torch.zeros(1, dtype=torch.long, device=device))
    motion_len = torch.minimum(motion_len_a, motion_len_b)
    loop_mode_a = motion_a.get_motion_loop_mode(torch.zeros(1, dtype=torch.long, device=device))
    loop_mode_b = motion_b.get_motion_loop_mode(torch.zeros(1, dtype=torch.long, device=device))
    can_wrap = (
        loop_mode_a.item() == motion.LoopMode.WRAP.value
        and loop_mode_b.item() == motion.LoopMode.WRAP.value
    )

    t = 0.0
    dt = sim_engine.get_timestep()

    try:
        while True:
            motion_time = torch.tensor([t], dtype=torch.float32, device=device)

            if motion_time.item() > motion_len.item():
                if repeat:
                    t = 0.0
                    motion_time[:] = 0.0
                else:
                    break

            root_pos_a, root_rot_a, _, _, joint_rot_a, _ = sample_motion(motion_a, motion_time)
            root_pos_b, root_rot_b, _, _, joint_rot_b, _ = sample_motion(motion_b, motion_time)
            joint_dof_a = motion_a.joint_rot_to_dof(joint_rot_a)
            joint_dof_b = motion_b.joint_rot_to_dof(joint_rot_b)

            set_char_pose(sim_engine, char_model, char_a, root_pos_a, root_rot_a, joint_rot_a, joint_dof_a)
            set_char_pose(sim_engine, char_model, char_b, root_pos_b, root_rot_b, joint_rot_b, joint_dof_b)

            midpoint = 0.5 * (root_pos_a[0].cpu().numpy() + root_pos_b[0].cpu().numpy())
            cam_pos = np.array(
                [
                    midpoint[0],
                    midpoint[1] - args.camera_distance,
                    midpoint[2] + args.camera_height,
                ]
            )
            cam_target = np.array([midpoint[0], midpoint[1], midpoint[2] + 0.8])
            sim_engine.set_camera_pose(cam_pos, cam_target)
            sim_engine.render()

            t += dt * args.time_scale
            if args.sleep > 0.0:
                time.sleep(args.sleep)
            elif can_wrap and motion_time.item() > motion_len.item():
                t = 0.0
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
