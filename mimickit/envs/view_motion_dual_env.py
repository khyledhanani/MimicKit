import os

import gymnasium.spaces as spaces
import numpy as np
import torch

import anim.motion as motion
import anim.motion_lib as motion_lib
import engines.engine as engine
import envs.base_env as base_env
import envs.char_env as char_env
import envs.sim_env as sim_env
import util.camera as camera
import util.torch_util as torch_util
from util.logger import Logger


class ViewMotionDualEnv(sim_env.SimEnv):
    """Play two MimicKit motion clips in parallel (one env slot per view); mirrors ``view_motion`` + ``tools/view_paired_motion``."""

    def __init__(self, env_config, engine_config, num_envs, device, visualize):
        self._time_scale = float(env_config.get("time_scale", 1.0))
        engine_config["sim_freq"] = engine_config["control_freq"]

        self._global_obs = env_config["global_obs"]
        self._root_height_obs = env_config.get("root_height_obs", True)

        super().__init__(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=False,
        )

        self._print_char_prop(0, self._char_a_ids[0], "A")
        self._print_char_prop(0, self._char_b_ids[0], "B")
        self._validate_envs(self._char_a_ids[0], "A", self._kin_char_model_a)
        self._validate_envs(self._char_b_ids[0], "B", self._kin_char_model_b)

    def _build_kin_char_model_for_file(self, char_file):
        _, file_ext = os.path.splitext(char_file)
        if file_ext == ".xml":
            import anim.mjcf_char_model as mjcf_char_model

            char_model = mjcf_char_model.MJCFCharModel(self._device)
        elif file_ext == ".urdf":
            import anim.urdf_char_model as urdf_char_model

            char_model = urdf_char_model.URDFCharModel(self._device)
        elif file_ext == ".usd":
            import anim.usd_char_model as usd_char_model

            char_model = usd_char_model.USDCharModel(self._device)
        else:
            assert False, "Unsupported character file format: {:s}".format(file_ext)

        char_model.load(char_file)
        return char_model

    def _parse_init_pose_side(self, init_pose, kin_char_model):
        dof_size = kin_char_model.get_dof_size()
        if init_pose is None:
            init_pose = torch.zeros(6 + dof_size, dtype=torch.float32, device=self._device)
        else:
            init_pose = torch.tensor(init_pose, dtype=torch.float32, device=self._device)
            if init_pose.shape[-1] == 3:
                init_pose = torch.cat([init_pose, torch.zeros(3 + dof_size, device=self._device)], dim=-1)

        init_root_pos, init_root_rot, init_dof_pos = motion_lib.extract_pose_data(init_pose)
        init_root_rot = torch_util.exp_map_to_quat(init_root_rot)
        return init_root_pos, init_root_rot, init_dof_pos

    def _build_envs(self, env_config, num_envs):
        shared_char = env_config.get("char_file", None)
        self._char_file_a = env_config.get("char_file_a", shared_char)
        self._char_file_b = env_config.get("char_file_b", shared_char)
        assert self._char_file_a is not None, "Set char_file or char_file_a"
        assert self._char_file_b is not None, "Set char_file or char_file_b"

        self._kin_char_model_a = self._build_kin_char_model_for_file(self._char_file_a)
        self._kin_char_model_b = self._build_kin_char_model_for_file(self._char_file_b)

        shared_init = env_config.get("init_pose", None)
        self._init_root_pos_a, self._init_root_rot_a, self._init_dof_pos_a = self._parse_init_pose_side(
            env_config.get("init_pose_a", shared_init), self._kin_char_model_a
        )
        self._init_root_pos_b, self._init_root_rot_b, self._init_dof_pos_b = self._parse_init_pose_side(
            env_config.get("init_pose_b", shared_init), self._kin_char_model_b
        )

        self._char_a_ids = []
        self._char_b_ids = []

        for e in range(num_envs):
            Logger.print("Building {:d}/{:d} envs".format(e + 1, num_envs), end="\r")
            env_id = self._engine.create_env()
            assert env_id == e
            self._build_env(env_id, env_config)
        Logger.print("\n")

        self._motion_lib_a = motion_lib.MotionLib(
            motion_file=env_config["motion_file_a"], kin_char_model=self._kin_char_model_a, device=self._device
        )
        self._motion_lib_b = motion_lib.MotionLib(
            motion_file=env_config["motion_file_b"], kin_char_model=self._kin_char_model_b, device=self._device
        )

    def _build_env(self, env_id, env_config):
        col_a = np.array([0.25, 0.55, 0.95])
        col_b = np.array([0.35, 0.80, 0.35])
        start_pos_a = self._init_root_pos_a.cpu().numpy()
        start_rot_a = self._init_root_rot_a.cpu().numpy()
        start_pos_b = self._init_root_pos_b.cpu().numpy()
        start_rot_b = self._init_root_rot_b.cpu().numpy()

        a_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.articulated,
            asset_file=self._char_file_a,
            name="character_a",
            is_visual=True,
            enable_self_collisions=False,
            start_pos=start_pos_a,
            start_rot=start_rot_a,
            disable_motors=True,
            color=col_a,
        )
        b_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.articulated,
            asset_file=self._char_file_b,
            name="character_b",
            is_visual=True,
            enable_self_collisions=False,
            start_pos=start_pos_b,
            start_rot=start_rot_b,
            disable_motors=True,
            color=col_b,
        )

        if env_id == 0:
            self._char_a_ids.append(a_id)
            self._char_b_ids.append(b_id)
        else:
            assert self._char_a_ids[0] == a_id
            assert self._char_b_ids[0] == b_id

    def _build_action_space(self):
        control_mode = self._engine.get_control_mode()
        char_a = self._char_a_ids[0]
        if control_mode == engine.ControlMode.none:
            action_size = int(self._engine.get_dof_pos(char_a).shape[-1])
            low = -np.ones(action_size, dtype=np.float32)
            high = np.ones(action_size, dtype=np.float32)
        elif control_mode == engine.ControlMode.vel:
            action_size = int(self._engine.get_dof_pos(char_a).shape[-1])
            low = -2.0 * np.pi * np.ones(action_size, dtype=np.float32)
            high = 2.0 * np.pi * np.ones(action_size, dtype=np.float32)
        elif control_mode == engine.ControlMode.torque:
            torque_lim = self._engine.get_obj_torque_limits(0, char_a)
            low = -np.array(torque_lim, dtype=np.float32)
            high = np.array(torque_lim, dtype=np.float32)
        elif control_mode in (engine.ControlMode.pos, engine.ControlMode.pd_explicit):
            dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_a)
            low, high = self._build_pos_bounds(dof_low, dof_high)
        else:
            assert False, "Unsupported control mode: {}".format(control_mode)

        return spaces.Box(low=low, high=high)

    def _build_pos_bounds(self, dof_low, dof_high):
        low = np.zeros(dof_high.shape, dtype=np.float32)
        high = np.zeros(dof_high.shape, dtype=np.float32)
        num_joints = self._kin_char_model_a.get_num_joints()
        for j in range(1, num_joints):
            curr_joint = self._kin_char_model_a.get_joint(j)
            j_dof_dim = curr_joint.get_dof_dim()
            if j_dof_dim <= 0:
                continue
            if j_dof_dim == 3:
                j_low = curr_joint.get_joint_dof(dof_low)
                j_high = curr_joint.get_joint_dof(dof_high)
                curr_scale = 1.2 * max(np.max(np.abs(j_low)), np.max(np.abs(j_high)))
                curr_low, curr_high = -curr_scale, curr_scale
            else:
                j_low = curr_joint.get_joint_dof(dof_low)
                j_high = curr_joint.get_joint_dof(dof_high)
                curr_mid = 0.5 * (j_high + j_low)
                curr_scale = 1.4 * np.maximum(np.abs(j_high - curr_mid), np.abs(j_low - curr_mid))
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale
            curr_joint.set_joint_dof(curr_low, low)
            curr_joint.set_joint_dof(curr_high, high)
        return low, high

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        self._action_bound_low = torch.tensor(self._action_space.low, device=self._device)
        self._action_bound_high = torch.tensor(self._action_space.high, device=self._device)

        key_bodies = env_config.get("key_bodies", [])
        self._key_body_ids = self._build_body_ids_tensor(self._char_a_ids[0], key_bodies)
        if self._has_key_bodies():
            body_pos = self._engine.get_body_pos(self._char_a_ids[0])
            self._ref_body_pos = torch.zeros_like(body_pos)

    def _build_body_ids_tensor(self, char_id, body_names):
        body_ids = []
        for body_name in body_names:
            body_id = self._engine.find_obj_body_id(char_id, body_name)
            assert body_id != -1
            body_ids.append(body_id)
        return torch.tensor(body_ids, device=self._device, dtype=torch.long)

    def _has_key_bodies(self):
        return len(self._key_body_ids) > 0

    def _print_char_prop(self, env_id, obj_id, label):
        num_dofs = self._engine.get_obj_num_dofs(obj_id)
        total_mass = self._engine.calc_obj_mass(env_id, obj_id)
        Logger.print(
            "Char {} {:d} properties\n\tDoFs: {:d}\n\tMass: {:.3f} kg\n".format(label, obj_id, num_dofs, total_mass)
        )

    def _validate_envs(self, char_id, label, kin_model):
        sim_body_names = self._engine.get_obj_body_names(char_id)
        kin_body_names = kin_model.get_body_names()
        for sim_name, kin_name in zip(sim_body_names, kin_body_names):
            assert sim_name == kin_name, "Char {} body mismatch: {} vs {}".format(label, sim_name, kin_name)

    def _build_camera(self, env_config):
        env_id = 0
        char_id = self._char_a_ids[0]
        char_root_pos = self._engine.get_root_pos(char_id)
        char_pos = char_root_pos[env_id].cpu().numpy()
        cam_pos = np.array([char_pos[0], char_pos[1] - 5.0, 3.0])
        cam_target = np.array([char_pos[0], char_pos[1], 1.0])
        cam_mode = camera.CameraMode[env_config["camera_mode"]]
        self._camera = camera.Camera(
            mode=cam_mode,
            engine=self._engine,
            pos=cam_pos,
            target=cam_target,
            track_env_id=env_id,
            track_obj_id=char_id,
        )

    def _get_env_motion_ids_a(self):
        n = self._motion_lib_a.get_num_motions()
        return torch.remainder(self._env_ids, n)

    def _get_env_motion_ids_b(self):
        n = self._motion_lib_b.get_num_motions()
        return torch.remainder(self._env_ids, n)

    def _apply_action(self, actions):
        return

    def _sync_motion(self):
        motion_ids_a = self._get_env_motion_ids_a()
        motion_ids_b = self._get_env_motion_ids_b()
        motion_times = self._time_buf * self._time_scale

        root_pos_a, root_rot_a, _, _, joint_rot_a, _ = self._motion_lib_a.calc_motion_frame(motion_ids_a, motion_times)
        root_pos_b, root_rot_b, _, _, joint_rot_b, _ = self._motion_lib_b.calc_motion_frame(motion_ids_b, motion_times)

        joint_dof_a = self._motion_lib_a.joint_rot_to_dof(joint_rot_a)
        joint_dof_b = self._motion_lib_b.joint_rot_to_dof(joint_rot_b)

        char_a = self._char_a_ids[0]
        char_b = self._char_b_ids[0]

        self._engine.set_root_pos(None, char_a, root_pos_a)
        self._engine.set_root_rot(None, char_a, root_rot_a)
        self._engine.set_root_vel(None, char_a, 0.0)
        self._engine.set_root_ang_vel(None, char_a, 0.0)
        self._engine.set_dof_pos(None, char_a, joint_dof_a)
        self._engine.set_dof_vel(None, char_a, 0.0)

        self._engine.set_root_pos(None, char_b, root_pos_b)
        self._engine.set_root_rot(None, char_b, root_rot_b)
        self._engine.set_root_vel(None, char_b, 0.0)
        self._engine.set_root_ang_vel(None, char_b, 0.0)
        self._engine.set_dof_pos(None, char_b, joint_dof_b)
        self._engine.set_dof_vel(None, char_b, 0.0)

        body_pos_a, body_rot_a = self._kin_char_model_a.forward_kinematics(
            root_pos=root_pos_a, root_rot=root_rot_a, joint_rot=joint_rot_a
        )
        body_pos_b, body_rot_b = self._kin_char_model_b.forward_kinematics(
            root_pos=root_pos_b, root_rot=root_rot_b, joint_rot=joint_rot_b
        )

        if self._has_key_bodies():
            self._ref_body_pos[:] = body_pos_a

        self._engine.set_body_pos(None, char_a, body_pos_a)
        self._engine.set_body_rot(None, char_a, body_rot_a)
        self._engine.set_body_pos(None, char_b, body_pos_b)
        self._engine.set_body_rot(None, char_b, body_rot_b)

    def _update_misc(self):
        super()._update_misc()
        self._sync_motion()

    def _update_reward(self):
        char_id = self._char_a_ids[0]
        char_root_pos = self._engine.get_root_pos(char_id)
        self._reward_buf[:] = char_env.compute_reward(char_root_pos)

    def _update_done(self):
        motion_ids_a = self._get_env_motion_ids_a()
        motion_ids_b = self._get_env_motion_ids_b()
        la = self._motion_lib_a.get_motion_length(motion_ids_a)
        lb = self._motion_lib_b.get_motion_length(motion_ids_b)
        motion_len_min = torch.minimum(la, lb)
        ma = self._motion_lib_a.get_motion_loop_mode(motion_ids_a)
        mb = self._motion_lib_b.get_motion_loop_mode(motion_ids_b)
        both_wrap = (ma == motion.LoopMode.WRAP.value) & (mb == motion.LoopMode.WRAP.value)
        self._done_buf[:] = compute_done_dual(self._done_buf, self._time_buf, motion_len_min, both_wrap)

    def _compute_obs(self, env_ids=None):
        char_id = self._char_a_ids[0]
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]

        joint_rot = self._kin_char_model_a.dof_to_rot(dof_pos)
        if self._has_key_bodies():
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        return char_env.compute_char_obs(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_obs=self._global_obs,
            root_height_obs=self._root_height_obs,
        )

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) == 0:
            return

        char_a = self._char_a_ids[0]
        char_b = self._char_b_ids[0]

        self._engine.set_root_pos(env_ids, char_a, self._init_root_pos_a)
        self._engine.set_root_rot(env_ids, char_a, self._init_root_rot_a)
        self._engine.set_root_vel(env_ids, char_a, 0.0)
        self._engine.set_root_ang_vel(env_ids, char_a, 0.0)
        self._engine.set_dof_pos(env_ids, char_a, self._init_dof_pos_a)
        self._engine.set_dof_vel(env_ids, char_a, 0.0)
        self._engine.set_body_vel(env_ids, char_a, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_a, 0.0)

        self._engine.set_root_pos(env_ids, char_b, self._init_root_pos_b)
        self._engine.set_root_rot(env_ids, char_b, self._init_root_rot_b)
        self._engine.set_root_vel(env_ids, char_b, 0.0)
        self._engine.set_root_ang_vel(env_ids, char_b, 0.0)
        self._engine.set_dof_pos(env_ids, char_b, self._init_dof_pos_b)
        self._engine.set_dof_vel(env_ids, char_b, 0.0)
        self._engine.set_body_vel(env_ids, char_b, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_b, 0.0)

        root_pos = self._engine.get_root_pos(char_a)[env_ids]
        root_rot = self._engine.get_root_rot(char_a)[env_ids]
        dof_pos = self._engine.get_dof_pos(char_a)[env_ids]
        joint_rot = self._kin_char_model_a.dof_to_rot(dof_pos)
        body_pos, body_rot = self._kin_char_model_a.forward_kinematics(root_pos, root_rot, joint_rot)
        self._engine.set_body_pos(env_ids, char_a, body_pos)
        self._engine.set_body_rot(env_ids, char_a, body_rot)

        root_pos_b = self._engine.get_root_pos(char_b)[env_ids]
        root_rot_b = self._engine.get_root_rot(char_b)[env_ids]
        dof_pos_b = self._engine.get_dof_pos(char_b)[env_ids]
        joint_rot_b = self._kin_char_model_b.dof_to_rot(dof_pos_b)
        body_pos_b, body_rot_b = self._kin_char_model_b.forward_kinematics(root_pos_b, root_rot_b, joint_rot_b)
        self._engine.set_body_pos(env_ids, char_b, body_pos_b)
        self._engine.set_body_rot(env_ids, char_b, body_rot_b)

    def _render_scene(self):
        super()._render_scene()
        self._render_key_points()

    def _render_key_points(self):
        if not self._has_key_bodies():
            return
        line_width = 2.0
        num_key_bodies = self._key_body_ids.shape[0]
        cols = np.array(3 * num_key_bodies * [[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        num_envs = self.get_num_envs()
        for i in range(num_envs):
            key_body_pos = self._ref_body_pos[i][self._key_body_ids].cpu().numpy()
            start_verts = 0.2 * np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
            end_verts = 0.2 * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            key_body_pos = np.expand_dims(key_body_pos, -2)
            start_verts = np.expand_dims(start_verts, 0) + key_body_pos
            end_verts = np.expand_dims(end_verts, 0) + key_body_pos
            start_verts = start_verts.reshape([-1, 3])
            end_verts = end_verts.reshape([-1, 3])
            self._engine.draw_lines(i, start_verts, end_verts, cols, line_width)


@torch.jit.script
def compute_done_dual(done_buf, time, motion_len_min, both_wrap):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    num_loops = 5
    end_time = motion_len_min.clone()
    end_time[both_wrap] = end_time[both_wrap] * num_loops
    timeout = time >= end_time
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    done[timeout] = base_env.DoneFlags.TIME.value
    return done
