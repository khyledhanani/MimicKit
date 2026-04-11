import os
import gymnasium.spaces as spaces
import numpy as np
import torch

import anim.motion as motion
import anim.motion_lib as motion_lib
import envs.base_env as base_env
import envs.char_env as char_env
import envs.deepmimic_env as deepmimic_env
import envs.sim_env as sim_env
import engines.engine as engine
import util.camera as camera
import util.torch_util as torch_util
from util.logger import Logger


class DualDeepMimicEnv(sim_env.SimEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._enable_early_termination = env_config.get("enable_early_termination", True)
        self._pose_termination = env_config.get("pose_termination", False)
        self._pose_termination_dist = env_config.get("pose_termination_dist", 1.0)
        self._global_obs = env_config["global_obs"]
        self._root_height_obs = env_config.get("root_height_obs", True)
        self._rand_reset = env_config.get("rand_reset", True)
        self._enable_phase_obs = env_config.get("enable_phase_obs", True)
        self._enable_tar_obs = env_config.get("enable_tar_obs", False)
        self._tar_obs_steps = env_config.get("tar_obs_steps", [1])
        self._tar_obs_steps = torch.tensor(self._tar_obs_steps, device=device, dtype=torch.int)
        self._num_phase_encoding = env_config.get("num_phase_encoding", 0)
        self._track_root_h = env_config.get("track_root_h", True)
        self._track_root = env_config.get("track_root", True)
        self._control_mode = env_config.get("control_mode", "dual")
        self._enable_self_collisions = env_config.get("enable_self_collisions", False)
        self._visual_reference_partner = env_config.get("visual_reference_partner", True)
        self._ground_align_on_reset = env_config.get("ground_align_on_reset", False)
        assert self._control_mode in ("dual", "single_a", "single_b"), "control_mode must be dual, single_a, or single_b"

        self._inter_actor_collisions = env_config.get("inter_actor_collisions", False)
        self._inter_actor_contact_bodies_a = env_config.get("inter_actor_contact_bodies_a", None)
        self._inter_actor_contact_bodies_b = env_config.get("inter_actor_contact_bodies_b", None)
        self._reward_contact_w = env_config.get("reward_contact_w", 0.0)
        self._reward_contact_scale = env_config.get("reward_contact_scale", 10.0)
        self._contact_dist_threshold = env_config.get("contact_dist_threshold", 0.1)

        if self._inter_actor_collisions:
            assert self._control_mode == "dual", \
                "inter_actor_collisions requires control_mode=dual so both actors are dynamic"

        self._reward_pose_w = env_config.get("reward_pose_w", 0.5)
        self._reward_vel_w = env_config.get("reward_vel_w", 0.1)
        self._reward_root_pose_w = env_config.get("reward_root_pose_w", 0.15)
        self._reward_root_vel_w = env_config.get("reward_root_vel_w", 0.1)
        self._reward_key_pos_w = env_config.get("reward_key_pos_w", 0.15)
        self._reward_rel_key_pos_w = env_config.get("reward_rel_key_pos_w", 0.2)

        self._reward_pose_scale = env_config.get("reward_pose_scale", 0.25)
        self._reward_vel_scale = env_config.get("reward_vel_scale", 0.01)
        self._reward_root_pose_scale = env_config.get("reward_root_pose_scale", 5.0)
        self._reward_root_vel_scale = env_config.get("reward_root_vel_scale", 1.0)
        self._reward_key_pos_scale = env_config.get("reward_key_pos_scale", 10.0)
        self._reward_rel_key_pos_scale = env_config.get("reward_rel_key_pos_scale", 10.0)

        self._reward_balance_w = env_config.get("reward_balance_w", 0.0)
        self._reward_upright_w = env_config.get("reward_upright_w", 0.0)
        self._reward_foot_contact_w = env_config.get("reward_foot_contact_w", 0.0)
        self._target_root_height = env_config.get("target_root_height", 0.9)

        super().__init__(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )

        self._print_char_prop(0, self._char_a_ids[0], "A")
        self._print_char_prop(0, self._char_b_ids[0], "B")
        self._validate_envs()

    def _get_char_file(self, env_config, side):
        side = side.lower()
        shared_char_file = env_config.get("char_file", None)
        return env_config.get(f"char_file_{side}", shared_char_file)

    def _get_side_body_names(self, env_config, base_key, side):
        side_key = f"{base_key}_{side.lower()}"
        return env_config.get(side_key, env_config.get(base_key, []))

    def _build_kin_char_model_for_file(self, char_file):
        _, file_ext = os.path.splitext(char_file)
        if file_ext == ".xml":
            import anim.mjcf_char_model as mjcf_char_model

            model = mjcf_char_model.MJCFCharModel(self._device)
        elif file_ext == ".urdf":
            import anim.urdf_char_model as urdf_char_model

            model = urdf_char_model.URDFCharModel(self._device)
        elif file_ext == ".usd":
            import anim.usd_char_model as usd_char_model

            model = usd_char_model.USDCharModel(self._device)
        else:
            raise ValueError("Unsupported character file format: {}".format(file_ext))
        model.load(char_file)
        return model

    def _parse_init_pose(self, init_pose, kin_char_model, device):
        dof_size = kin_char_model.get_dof_size()
        if init_pose is None:
            init_pose = torch.zeros(6 + dof_size, dtype=torch.float32, device=device)
        else:
            init_pose = torch.tensor(init_pose, dtype=torch.float32, device=device)
            if init_pose.shape[-1] == 3:
                init_pose = torch.cat([init_pose, torch.zeros(3 + dof_size, device=device)], dim=-1)

        init_root_pos, init_root_rot, init_dof_pos = motion_lib.extract_pose_data(init_pose)
        init_root_rot = torch_util.exp_map_to_quat(init_root_rot)
        return init_root_pos, init_root_rot, init_dof_pos

    def _parse_joint_err_weights_for_model(self, kin_char_model, joint_err_w):
        num_joints = kin_char_model.get_num_joints()
        if joint_err_w is None:
            joint_err_w = torch.ones(num_joints - 1, device=self._device, dtype=torch.float32)
        else:
            joint_err_w = torch.tensor(joint_err_w, device=self._device, dtype=torch.float32)

        assert joint_err_w.shape[-1] == num_joints - 1

        dof_size = kin_char_model.get_dof_size()
        dof_err_w = torch.zeros(dof_size, device=self._device, dtype=torch.float32)
        for j in range(1, num_joints):
            dim = kin_char_model.get_joint_dof_dim(j)
            if dim > 0:
                idx = kin_char_model.get_joint_dof_idx(j)
                dof_err_w[idx : idx + dim] = joint_err_w[j - 1]

        return joint_err_w, dof_err_w

    def _build_action_bounds_single(self, char_id, kin_char_model, control_mode):
        if control_mode == engine.ControlMode.none:
            action_size = int(self._engine.get_dof_pos(char_id).shape[-1])
            low = -np.ones(action_size, dtype=np.float32)
            high = np.ones(action_size, dtype=np.float32)
        elif control_mode == engine.ControlMode.vel:
            action_size = int(self._engine.get_dof_pos(char_id).shape[-1])
            low = -2.0 * np.pi * np.ones(action_size, dtype=np.float32)
            high = 2.0 * np.pi * np.ones(action_size, dtype=np.float32)
        elif control_mode == engine.ControlMode.torque:
            torque_lim = self._engine.get_obj_torque_limits(0, char_id)
            low = -np.array(torque_lim, dtype=np.float32)
            high = np.array(torque_lim, dtype=np.float32)
        elif control_mode in (engine.ControlMode.pos, engine.ControlMode.pd_explicit):
            dof_low, dof_high = self._engine.get_obj_dof_limits(0, char_id)
            low, high = self._build_pos_bounds_single(dof_low, dof_high, kin_char_model)
        else:
            raise ValueError("Unsupported control mode: {}".format(control_mode))
        return low, high

    def _build_envs(self, env_config, num_envs):
        self._char_file_a = self._get_char_file(env_config, "a")
        self._char_file_b = self._get_char_file(env_config, "b")
        assert self._char_file_a is not None, "Missing char_file or char_file_a"
        assert self._char_file_b is not None, "Missing char_file or char_file_b"

        self._kin_char_model_a = self._build_kin_char_model_for_file(self._char_file_a)
        self._kin_char_model_b = self._build_kin_char_model_for_file(self._char_file_b)

        self._init_root_pos_a, self._init_root_rot_a, self._init_dof_pos_a = self._parse_init_pose(
            env_config.get("init_pose_a", env_config.get("init_pose", None)), self._kin_char_model_a, self._device
        )
        self._init_root_pos_b, self._init_root_rot_b, self._init_dof_pos_b = self._parse_init_pose(
            env_config.get("init_pose_b", env_config.get("init_pose", None)), self._kin_char_model_b, self._device
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

        # Must be called before engine.initialize_sim()
        if self._inter_actor_collisions:
            self._engine.configure_inter_actor_collisions(
                inter_actor_collisions=True,
                bodies_a=self._inter_actor_contact_bodies_a,
                bodies_b=self._inter_actor_contact_bodies_b,
            )

    def _build_env(self, env_id, env_config):
        col_a = np.array([0.25, 0.55, 0.95])
        col_b = np.array([0.35, 0.80, 0.35])
        a_is_visual = self._visual_reference_partner and self._control_mode == "single_b"
        b_is_visual = self._visual_reference_partner and self._control_mode == "single_a"
        a_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.articulated,
            asset_file=self._char_file_a,
            name="character_a",
            is_visual=a_is_visual,
            enable_self_collisions=self._enable_self_collisions,
            start_pos=self._init_root_pos_a.cpu().numpy(),
            start_rot=self._init_root_rot_a.cpu().numpy(),
            disable_motors=a_is_visual,
            color=col_a,
        )
        b_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.articulated,
            asset_file=self._char_file_b,
            name="character_b",
            is_visual=b_is_visual,
            enable_self_collisions=self._enable_self_collisions,
            start_pos=self._init_root_pos_b.cpu().numpy(),
            start_rot=self._init_root_rot_b.cpu().numpy(),
            disable_motors=b_is_visual,
            color=col_b,
        )

        if env_id == 0:
            self._char_a_ids.append(a_id)
            self._char_b_ids.append(b_id)
        else:
            assert self._char_a_ids[0] == a_id
            assert self._char_b_ids[0] == b_id

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        n = self.get_num_envs()
        char_a = self._char_a_ids[0]
        char_b = self._char_b_ids[0]
        root_pos_a = self._engine.get_root_pos(char_a)
        root_rot_a = self._engine.get_root_rot(char_a)
        root_vel_a = self._engine.get_root_vel(char_a)
        root_ang_vel_a = self._engine.get_root_ang_vel(char_a)
        body_pos_a = self._engine.get_body_pos(char_a)
        body_rot_a = self._engine.get_body_rot(char_a)
        dof_pos_a = self._engine.get_dof_pos(char_a)
        dof_vel_a = self._engine.get_dof_vel(char_a)
        root_pos_b = self._engine.get_root_pos(char_b)
        root_rot_b = self._engine.get_root_rot(char_b)
        root_vel_b = self._engine.get_root_vel(char_b)
        root_ang_vel_b = self._engine.get_root_ang_vel(char_b)
        body_pos_b = self._engine.get_body_pos(char_b)
        body_rot_b = self._engine.get_body_rot(char_b)
        dof_pos_b = self._engine.get_dof_pos(char_b)
        dof_vel_b = self._engine.get_dof_vel(char_b)

        self._motion_ids = torch.zeros(n, device=self._device, dtype=torch.long)
        self._motion_time_offsets = torch.zeros(n, device=self._device, dtype=torch.float32)

        self._ref_root_pos_a = torch.zeros_like(root_pos_a)
        self._ref_root_rot_a = torch.zeros_like(root_rot_a)
        self._ref_root_vel_a = torch.zeros_like(root_vel_a)
        self._ref_root_ang_vel_a = torch.zeros_like(root_ang_vel_a)
        self._ref_body_pos_a = torch.zeros_like(body_pos_a)
        self._ref_body_rot_a = torch.zeros_like(body_rot_a)
        self._ref_joint_rot_a = torch.zeros_like(body_rot_a[..., 1:, :])
        self._ref_dof_pos_a = torch.zeros_like(dof_pos_a)
        self._ref_dof_vel_a = torch.zeros_like(dof_vel_a)

        self._ref_root_pos_b = torch.zeros_like(root_pos_b)
        self._ref_root_rot_b = torch.zeros_like(root_rot_b)
        self._ref_root_vel_b = torch.zeros_like(root_vel_b)
        self._ref_root_ang_vel_b = torch.zeros_like(root_ang_vel_b)
        self._ref_body_pos_b = torch.zeros_like(body_pos_b)
        self._ref_body_rot_b = torch.zeros_like(body_rot_b)
        self._ref_joint_rot_b = torch.zeros_like(body_rot_b[..., 1:, :])
        self._ref_dof_pos_b = torch.zeros_like(dof_pos_b)
        self._ref_dof_vel_b = torch.zeros_like(dof_vel_b)

        key_bodies_a = self._get_side_body_names(env_config, "key_bodies", "a")
        key_bodies_b = self._get_side_body_names(env_config, "key_bodies", "b")
        self._key_body_ids_a = self._build_body_ids_tensor(self._char_a_ids[0], key_bodies_a)
        self._key_body_ids_b = self._build_body_ids_tensor(self._char_b_ids[0], key_bodies_b)
        if self._key_body_ids_a.numel() > 0 and self._key_body_ids_b.numel() > 0:
            assert self._key_body_ids_a.numel() == self._key_body_ids_b.numel(), \
                "key_bodies_a and key_bodies_b must have the same number of bodies"

        contact_bodies_a = self._get_side_body_names(env_config, "contact_bodies", "a")
        contact_bodies_b = self._get_side_body_names(env_config, "contact_bodies", "b")
        self._contact_body_ids_a = self._build_body_ids_tensor(self._char_a_ids[0], contact_bodies_a)
        self._contact_body_ids_b = self._build_body_ids_tensor(self._char_b_ids[0], contact_bodies_b)

        pose_termination_bodies_a = self._get_side_body_names(env_config, "pose_termination_bodies", "a")
        pose_termination_bodies_b = self._get_side_body_names(env_config, "pose_termination_bodies", "b")
        self._pose_termination_body_ids_a = self._build_body_ids_tensor(self._char_a_ids[0], pose_termination_bodies_a)
        self._pose_termination_body_ids_b = self._build_body_ids_tensor(self._char_b_ids[0], pose_termination_bodies_b)

        joint_err_w_a = env_config.get("joint_err_w_a", env_config.get("joint_err_w", None))
        joint_err_w_b = env_config.get("joint_err_w_b", env_config.get("joint_err_w", None))
        self._joint_err_w_a, self._dof_err_w_a = self._parse_joint_err_weights_for_model(self._kin_char_model_a, joint_err_w_a)
        self._joint_err_w_b, self._dof_err_w_b = self._parse_joint_err_weights_for_model(self._kin_char_model_b, joint_err_w_b)

        if self._inter_actor_collisions and self._inter_actor_contact_bodies_a:
            self._contact_body_ids_inter_a = self._build_body_ids_tensor(
                self._char_a_ids[0], self._inter_actor_contact_bodies_a
            )
            self._contact_body_ids_inter_b = self._build_body_ids_tensor(
                self._char_b_ids[0], self._inter_actor_contact_bodies_b
            )
        else:
            self._contact_body_ids_inter_a = torch.zeros(0, dtype=torch.long, device=self._device)
            self._contact_body_ids_inter_b = torch.zeros(0, dtype=torch.long, device=self._device)

        self._ground_align_dz_a = torch.zeros(n, device=self._device, dtype=torch.float32)
        self._ground_align_dz_b = torch.zeros(n, device=self._device, dtype=torch.float32)

    @staticmethod
    def _foot_ground_align_dz(body_pos, contact_body_ids):
        """World-space Δz so the lowest contact body rests on z=0 (negative of min foot z)."""
        if contact_body_ids.numel() == 0:
            return torch.zeros(body_pos.shape[0], device=body_pos.device, dtype=body_pos.dtype)
        z = body_pos[:, contact_body_ids, 2]
        return -z.min(dim=-1).values

    def _recompute_ground_align_dz_from_ref(self, env_ids):
        self._ground_align_dz_a[env_ids] = self._foot_ground_align_dz(
            self._ref_body_pos_a[env_ids], self._contact_body_ids_a
        )
        self._ground_align_dz_b[env_ids] = self._foot_ground_align_dz(
            self._ref_body_pos_b[env_ids], self._contact_body_ids_b
        )

    def _apply_ground_align_to_ref(self, env_ids=None):
        if not self._ground_align_on_reset:
            return
        if env_ids is None:
            self._ref_root_pos_a[:, 2] += self._ground_align_dz_a
            self._ref_root_pos_b[:, 2] += self._ground_align_dz_b
            self._ref_body_pos_a[:, :, 2] += self._ground_align_dz_a.unsqueeze(-1)
            self._ref_body_pos_b[:, :, 2] += self._ground_align_dz_b.unsqueeze(-1)
        else:
            da = self._ground_align_dz_a[env_ids]
            db = self._ground_align_dz_b[env_ids]
            self._ref_root_pos_a[env_ids, 2] += da
            self._ref_root_pos_b[env_ids, 2] += db
            self._ref_body_pos_a[env_ids, :, 2] += da.unsqueeze(-1)
            self._ref_body_pos_b[env_ids, :, 2] += db.unsqueeze(-1)

    def _build_action_space(self):
        control_mode = self._engine.get_control_mode()
        char_a = self._char_a_ids[0]
        char_b = self._char_b_ids[0]
        low_a, high_a = self._build_action_bounds_single(char_a, self._kin_char_model_a, control_mode)
        low_b, high_b = self._build_action_bounds_single(char_b, self._kin_char_model_b, control_mode)

        self._num_action_dof_a = int(low_a.shape[0])
        self._num_action_dof_b = int(low_b.shape[0])
        assert self._num_action_dof_a == self._kin_char_model_a.get_dof_size()
        assert self._num_action_dof_b == self._kin_char_model_b.get_dof_size()

        if self._control_mode == "dual":
            low = np.concatenate([low_a, low_b], axis=0)
            high = np.concatenate([high_a, high_b], axis=0)
        elif self._control_mode == "single_a":
            low = low_a
            high = high_a
        else:
            low = low_b
            high = high_b
        return spaces.Box(low=low, high=high)

    def _build_pos_bounds_single(self, dof_low, dof_high, kin_char_model):
        low = np.zeros(dof_high.shape, dtype=np.float32)
        high = np.zeros(dof_high.shape, dtype=np.float32)
        num_joints = kin_char_model.get_num_joints()
        for j in range(1, num_joints):
            joint = kin_char_model.get_joint(j)
            dim = joint.get_dof_dim()
            if dim <= 0:
                continue
            if dim == 3:
                j_low = joint.get_joint_dof(dof_low)
                j_high = joint.get_joint_dof(dof_high)
                scale = 1.2 * max(np.max(np.abs(j_low)), np.max(np.abs(j_high)))
                curr_low, curr_high = -scale, scale
            else:
                j_low = joint.get_joint_dof(dof_low)
                j_high = joint.get_joint_dof(dof_high)
                mid = 0.5 * (j_high + j_low)
                scale = 1.4 * np.maximum(np.abs(j_high - mid), np.abs(j_low - mid))
                curr_low = mid - scale
                curr_high = mid + scale
            joint.set_joint_dof(curr_low, low)
            joint.set_joint_dof(curr_high, high)
        return low, high

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._action_bound_low = torch.tensor(self._action_space.low, device=self._device)
        self._action_bound_high = torch.tensor(self._action_space.high, device=self._device)

    def _build_body_ids_tensor(self, char_id, body_names):
        ids = []
        for body_name in body_names:
            body_id = self._engine.find_obj_body_id(char_id, body_name)
            assert body_id != -1, "Body {} not found on character {}".format(body_name, char_id)
            ids.append(body_id)
        return torch.tensor(ids, device=self._device, dtype=torch.long)

    def _get_motion_times(self, env_ids=None):
        if env_ids is None:
            return self._time_buf + self._motion_time_offsets
        return self._time_buf[env_ids] + self._motion_time_offsets[env_ids]

    def _sample_motion_times(self, n):
        motion_ids = self._motion_lib_a.sample_motions(n)
        motion_ids = torch.remainder(motion_ids, self._motion_lib_b.get_num_motions())
        if self._rand_reset:
            motion_times = self._motion_lib_a.sample_time(motion_ids)
            motion_len_b = self._motion_lib_b.get_motion_length(motion_ids)
            motion_times = torch.minimum(motion_times, motion_len_b)
        else:
            motion_times = torch.zeros(n, dtype=torch.float32, device=self._device)
        return motion_ids, motion_times

    def _build_camera(self, env_config):
        root_a = self._engine.get_root_pos(self._char_a_ids[0])[0].cpu().numpy()
        root_b = self._engine.get_root_pos(self._char_b_ids[0])[0].cpu().numpy()
        mid = 0.5 * (root_a + root_b)
        cam_pos = np.array([mid[0], mid[1] - 4.0, 2.0])
        cam_target = np.array([mid[0], mid[1], 1.0])
        cam_mode = camera.CameraMode[env_config["camera_mode"]]
        self._camera = camera.Camera(
            mode=cam_mode,
            engine=self._engine,
            pos=cam_pos,
            target=cam_target,
            track_env_id=0,
            track_obj_id=self._char_a_ids[0],
        )

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) == 0:
            return
        motion_ids, motion_times = self._sample_motion_times(len(env_ids))
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        self._write_raw_ref_motion(env_ids)
        if self._ground_align_on_reset:
            self._recompute_ground_align_dz_from_ref(env_ids)
        else:
            self._ground_align_dz_a[env_ids] = 0
            self._ground_align_dz_b[env_ids] = 0
        self._apply_ground_align_to_ref(env_ids)
        self._sync_chars_to_ref(env_ids, sync_b=True)

    def _cache_engine_state(self):
        """Fetch engine state once per step and cache for obs/reward/done."""
        a_id = self._char_a_ids[0]
        b_id = self._char_b_ids[0]

        self._s_root_pos_a = self._engine.get_root_pos(a_id)
        self._s_root_rot_a = self._engine.get_root_rot(a_id)
        self._s_root_vel_a = self._engine.get_root_vel(a_id)
        self._s_root_ang_vel_a = self._engine.get_root_ang_vel(a_id)
        self._s_dof_pos_a = self._engine.get_dof_pos(a_id)
        self._s_dof_vel_a = self._engine.get_dof_vel(a_id)
        self._s_body_pos_a = self._engine.get_body_pos(a_id)

        self._s_root_pos_b = self._engine.get_root_pos(b_id)
        self._s_root_rot_b = self._engine.get_root_rot(b_id)
        self._s_root_vel_b = self._engine.get_root_vel(b_id)
        self._s_root_ang_vel_b = self._engine.get_root_ang_vel(b_id)
        self._s_dof_pos_b = self._engine.get_dof_pos(b_id)
        self._s_dof_vel_b = self._engine.get_dof_vel(b_id)
        self._s_body_pos_b = self._engine.get_body_pos(b_id)

        # Pre-compute joint rotations (used in both obs and reward)
        self._s_joint_rot_a = self._kin_char_model_a.dof_to_rot(self._s_dof_pos_a)
        self._s_joint_rot_b = self._kin_char_model_b.dof_to_rot(self._s_dof_pos_b)

        # Ground contact forces (used in reward and done)
        self._s_gcf_a = self._engine.get_ground_contact_forces(a_id)
        self._s_gcf_b = self._engine.get_ground_contact_forces(b_id)

    def _update_misc(self):
        self._cache_engine_state()
        self._update_ref_motion(None)
        if self._control_mode == "single_a":
            self._sync_char_b_to_ref(None)
        elif self._control_mode == "single_b":
            self._sync_char_a_to_ref(None)

    def _update_ref_motion(self, env_ids):
        self._write_raw_ref_motion(env_ids)
        self._apply_ground_align_to_ref(env_ids)

    def _write_raw_ref_motion(self, env_ids):
        if env_ids is None:
            ids = self._motion_ids
            times = self._get_motion_times()
            target = slice(None)
        else:
            ids = self._motion_ids[env_ids]
            times = self._get_motion_times(env_ids)
            target = env_ids

        a = self._motion_lib_a.calc_motion_frame(ids, times)
        b = self._motion_lib_b.calc_motion_frame(ids, times)

        root_pos_a, root_rot_a, root_vel_a, root_ang_vel_a, joint_rot_a, dof_vel_a = a
        root_pos_b, root_rot_b, root_vel_b, root_ang_vel_b, joint_rot_b, dof_vel_b = b

        body_pos_a, body_rot_a = self._kin_char_model_a.forward_kinematics(root_pos_a, root_rot_a, joint_rot_a)
        body_pos_b, body_rot_b = self._kin_char_model_b.forward_kinematics(root_pos_b, root_rot_b, joint_rot_b)

        self._ref_root_pos_a[target] = root_pos_a
        self._ref_root_rot_a[target] = root_rot_a
        self._ref_root_vel_a[target] = root_vel_a
        self._ref_root_ang_vel_a[target] = root_ang_vel_a
        self._ref_joint_rot_a[target] = joint_rot_a
        self._ref_dof_vel_a[target] = dof_vel_a
        self._ref_body_pos_a[target] = body_pos_a
        self._ref_body_rot_a[target] = body_rot_a
        self._ref_dof_pos_a[target] = self._motion_lib_a.joint_rot_to_dof(joint_rot_a)

        self._ref_root_pos_b[target] = root_pos_b
        self._ref_root_rot_b[target] = root_rot_b
        self._ref_root_vel_b[target] = root_vel_b
        self._ref_root_ang_vel_b[target] = root_ang_vel_b
        self._ref_joint_rot_b[target] = joint_rot_b
        self._ref_dof_vel_b[target] = dof_vel_b
        self._ref_body_pos_b[target] = body_pos_b
        self._ref_body_rot_b[target] = body_rot_b
        self._ref_dof_pos_b[target] = self._motion_lib_b.joint_rot_to_dof(joint_rot_b)

    def _sync_chars_to_ref(self, env_ids, sync_b):
        self._engine.set_root_pos(env_ids, self._char_a_ids[0], self._ref_root_pos_a[env_ids])
        self._engine.set_root_rot(env_ids, self._char_a_ids[0], self._ref_root_rot_a[env_ids])
        self._engine.set_root_vel(env_ids, self._char_a_ids[0], self._ref_root_vel_a[env_ids])
        self._engine.set_root_ang_vel(env_ids, self._char_a_ids[0], self._ref_root_ang_vel_a[env_ids])
        self._engine.set_dof_pos(env_ids, self._char_a_ids[0], self._ref_dof_pos_a[env_ids])
        self._engine.set_dof_vel(env_ids, self._char_a_ids[0], self._ref_dof_vel_a[env_ids])
        self._engine.set_body_pos(env_ids, self._char_a_ids[0], self._ref_body_pos_a[env_ids])
        self._engine.set_body_rot(env_ids, self._char_a_ids[0], self._ref_body_rot_a[env_ids])
        self._engine.set_body_vel(env_ids, self._char_a_ids[0], 0.0)
        self._engine.set_body_ang_vel(env_ids, self._char_a_ids[0], 0.0)
        if sync_b:
            self._sync_char_b_to_ref(env_ids)

    def _sync_char_b_to_ref(self, env_ids):
        if env_ids is None:
            val = slice(None)
        else:
            val = env_ids
        self._engine.set_root_pos(env_ids, self._char_b_ids[0], self._ref_root_pos_b[val])
        self._engine.set_root_rot(env_ids, self._char_b_ids[0], self._ref_root_rot_b[val])
        self._engine.set_root_vel(env_ids, self._char_b_ids[0], self._ref_root_vel_b[val])
        self._engine.set_root_ang_vel(env_ids, self._char_b_ids[0], self._ref_root_ang_vel_b[val])
        self._engine.set_dof_pos(env_ids, self._char_b_ids[0], self._ref_dof_pos_b[val])
        self._engine.set_dof_vel(env_ids, self._char_b_ids[0], self._ref_dof_vel_b[val])
        self._engine.set_body_pos(env_ids, self._char_b_ids[0], self._ref_body_pos_b[val])
        self._engine.set_body_rot(env_ids, self._char_b_ids[0], self._ref_body_rot_b[val])
        self._engine.set_body_vel(env_ids, self._char_b_ids[0], 0.0)
        self._engine.set_body_ang_vel(env_ids, self._char_b_ids[0], 0.0)

    def _sync_char_a_to_ref(self, env_ids):
        if env_ids is None:
            val = slice(None)
        else:
            val = env_ids
        self._engine.set_root_pos(env_ids, self._char_a_ids[0], self._ref_root_pos_a[val])
        self._engine.set_root_rot(env_ids, self._char_a_ids[0], self._ref_root_rot_a[val])
        self._engine.set_root_vel(env_ids, self._char_a_ids[0], self._ref_root_vel_a[val])
        self._engine.set_root_ang_vel(env_ids, self._char_a_ids[0], self._ref_root_ang_vel_a[val])
        self._engine.set_dof_pos(env_ids, self._char_a_ids[0], self._ref_dof_pos_a[val])
        self._engine.set_dof_vel(env_ids, self._char_a_ids[0], self._ref_dof_vel_a[val])
        self._engine.set_body_pos(env_ids, self._char_a_ids[0], self._ref_body_pos_a[val])
        self._engine.set_body_rot(env_ids, self._char_a_ids[0], self._ref_body_rot_a[val])
        self._engine.set_body_vel(env_ids, self._char_a_ids[0], 0.0)
        self._engine.set_body_ang_vel(env_ids, self._char_a_ids[0], 0.0)

    def _compute_obs(self, env_ids=None):
        has_cache = hasattr(self, '_s_root_pos_a')
        if env_ids is not None or not has_cache:
            # During reset or init: fetch directly from engine
            a_id = self._char_a_ids[0]
            b_id = self._char_b_ids[0]
            root_pos_a = self._engine.get_root_pos(a_id)
            root_rot_a = self._engine.get_root_rot(a_id)
            root_vel_a = self._engine.get_root_vel(a_id)
            root_ang_vel_a = self._engine.get_root_ang_vel(a_id)
            dof_pos_a = self._engine.get_dof_pos(a_id)
            dof_vel_a = self._engine.get_dof_vel(a_id)
            body_pos_a = self._engine.get_body_pos(a_id)
            root_pos_b = self._engine.get_root_pos(b_id)
            root_rot_b = self._engine.get_root_rot(b_id)
            root_vel_b = self._engine.get_root_vel(b_id)
            root_ang_vel_b = self._engine.get_root_ang_vel(b_id)
            dof_pos_b = self._engine.get_dof_pos(b_id)
            dof_vel_b = self._engine.get_dof_vel(b_id)
            body_pos_b = self._engine.get_body_pos(b_id)
            if env_ids is not None:
                root_pos_a = root_pos_a[env_ids]
                root_rot_a = root_rot_a[env_ids]
                root_vel_a = root_vel_a[env_ids]
                root_ang_vel_a = root_ang_vel_a[env_ids]
                dof_pos_a = dof_pos_a[env_ids]
                dof_vel_a = dof_vel_a[env_ids]
                body_pos_a = body_pos_a[env_ids]
                root_pos_b = root_pos_b[env_ids]
                root_rot_b = root_rot_b[env_ids]
                root_vel_b = root_vel_b[env_ids]
                root_ang_vel_b = root_ang_vel_b[env_ids]
                dof_pos_b = dof_pos_b[env_ids]
                dof_vel_b = dof_vel_b[env_ids]
                body_pos_b = body_pos_b[env_ids]
            joint_rot_a = self._kin_char_model_a.dof_to_rot(dof_pos_a)
            joint_rot_b = self._kin_char_model_b.dof_to_rot(dof_pos_b)
            if env_ids is not None:
                motion_ids = self._motion_ids[env_ids]
                motion_times = self._get_motion_times(env_ids)
            else:
                motion_ids = self._motion_ids
                motion_times = self._get_motion_times()
        else:
            # During step: use cached state
            root_pos_a = self._s_root_pos_a
            root_rot_a = self._s_root_rot_a
            root_vel_a = self._s_root_vel_a
            root_ang_vel_a = self._s_root_ang_vel_a
            dof_vel_a = self._s_dof_vel_a
            body_pos_a = self._s_body_pos_a
            root_pos_b = self._s_root_pos_b
            root_rot_b = self._s_root_rot_b
            root_vel_b = self._s_root_vel_b
            root_ang_vel_b = self._s_root_ang_vel_b
            dof_vel_b = self._s_dof_vel_b
            body_pos_b = self._s_body_pos_b
            joint_rot_a = self._s_joint_rot_a
            joint_rot_b = self._s_joint_rot_b
            motion_ids = self._motion_ids
            motion_times = self._get_motion_times()

        key_pos_a = body_pos_a[..., self._key_body_ids_a, :] if self._key_body_ids_a.numel() > 0 else torch.zeros([0], device=self._device)
        key_pos_b = body_pos_b[..., self._key_body_ids_b, :] if self._key_body_ids_b.numel() > 0 else torch.zeros([0], device=self._device)

        obs_a = char_env.compute_char_obs(
            root_pos_a, root_rot_a, root_vel_a, root_ang_vel_a, joint_rot_a, dof_vel_a, key_pos_a, self._global_obs, self._root_height_obs
        )
        obs_b = char_env.compute_char_obs(
            root_pos_b, root_rot_b, root_vel_b, root_ang_vel_b, joint_rot_b, dof_vel_b, key_pos_b, self._global_obs, self._root_height_obs
        )

        if self._enable_tar_obs:
            tar_root_pos_a, tar_root_rot_a, tar_joint_rot_a = self._fetch_tar_obs_data(
                self._motion_lib_a, motion_ids, motion_times
            )
            tar_root_pos_b, tar_root_rot_b, tar_joint_rot_b = self._fetch_tar_obs_data(
                self._motion_lib_b, motion_ids, motion_times
            )

            if self._ground_align_on_reset:
                if env_ids is not None:
                    dz_a = self._ground_align_dz_a[env_ids].unsqueeze(-1)
                    dz_b = self._ground_align_dz_b[env_ids].unsqueeze(-1)
                else:
                    dz_a = self._ground_align_dz_a.unsqueeze(-1)
                    dz_b = self._ground_align_dz_b.unsqueeze(-1)
                tar_root_pos_a = tar_root_pos_a.clone()
                tar_root_pos_b = tar_root_pos_b.clone()
                tar_root_pos_a[..., 2] += dz_a
                tar_root_pos_b[..., 2] += dz_b

            tar_root_pos_a_flat = torch.reshape(
                tar_root_pos_a, [tar_root_pos_a.shape[0] * tar_root_pos_a.shape[1], tar_root_pos_a.shape[-1]]
            )
            tar_root_rot_a_flat = torch.reshape(
                tar_root_rot_a, [tar_root_rot_a.shape[0] * tar_root_rot_a.shape[1], tar_root_rot_a.shape[-1]]
            )
            tar_joint_rot_a_flat = torch.reshape(
                tar_joint_rot_a,
                [tar_joint_rot_a.shape[0] * tar_joint_rot_a.shape[1], tar_joint_rot_a.shape[-2], tar_joint_rot_a.shape[-1]],
            )
            tar_body_pos_a_flat, _ = self._kin_char_model_a.forward_kinematics(
                tar_root_pos_a_flat, tar_root_rot_a_flat, tar_joint_rot_a_flat
            )
            tar_body_pos_a = torch.reshape(
                tar_body_pos_a_flat,
                [tar_root_pos_a.shape[0], tar_root_pos_a.shape[1], tar_body_pos_a_flat.shape[-2], tar_body_pos_a_flat.shape[-1]],
            )

            tar_root_pos_b_flat = torch.reshape(
                tar_root_pos_b, [tar_root_pos_b.shape[0] * tar_root_pos_b.shape[1], tar_root_pos_b.shape[-1]]
            )
            tar_root_rot_b_flat = torch.reshape(
                tar_root_rot_b, [tar_root_rot_b.shape[0] * tar_root_rot_b.shape[1], tar_root_rot_b.shape[-1]]
            )
            tar_joint_rot_b_flat = torch.reshape(
                tar_joint_rot_b,
                [tar_joint_rot_b.shape[0] * tar_joint_rot_b.shape[1], tar_joint_rot_b.shape[-2], tar_joint_rot_b.shape[-1]],
            )
            tar_body_pos_b_flat, _ = self._kin_char_model_b.forward_kinematics(
                tar_root_pos_b_flat, tar_root_rot_b_flat, tar_joint_rot_b_flat
            )
            tar_body_pos_b = torch.reshape(
                tar_body_pos_b_flat,
                [tar_root_pos_b.shape[0], tar_root_pos_b.shape[1], tar_body_pos_b_flat.shape[-2], tar_body_pos_b_flat.shape[-1]],
            )

            if self._key_body_ids_a.numel() > 0:
                tar_key_pos_a = tar_body_pos_a[..., self._key_body_ids_a, :]
            else:
                tar_key_pos_a = torch.zeros([0], device=self._device)

            if self._key_body_ids_b.numel() > 0:
                tar_key_pos_b = tar_body_pos_b[..., self._key_body_ids_b, :]
            else:
                tar_key_pos_b = torch.zeros([0], device=self._device)

            if self._global_obs:
                ref_root_pos_a = root_pos_a
                ref_root_rot_a = root_rot_a
                ref_root_pos_b = root_pos_b
                ref_root_rot_b = root_rot_b
            else:
                ref_root_pos_a = tar_root_pos_a[..., 0, :]
                ref_root_rot_a = tar_root_rot_a[..., 0, :]
                ref_root_pos_b = tar_root_pos_b[..., 0, :]
                ref_root_rot_b = tar_root_rot_b[..., 0, :]

            tar_obs_a = deepmimic_env.compute_tar_obs(
                ref_root_pos=ref_root_pos_a,
                ref_root_rot=ref_root_rot_a,
                root_pos=tar_root_pos_a,
                root_rot=tar_root_rot_a,
                joint_rot=tar_joint_rot_a,
                key_pos=tar_key_pos_a,
                global_obs=self._global_obs,
                root_height_obs=self._root_height_obs,
            )
            tar_obs_a = torch.reshape(tar_obs_a, [tar_obs_a.shape[0], tar_obs_a.shape[1] * tar_obs_a.shape[2]])
            obs_a = torch.cat([obs_a, tar_obs_a], dim=-1)

            tar_obs_b = deepmimic_env.compute_tar_obs(
                ref_root_pos=ref_root_pos_b,
                ref_root_rot=ref_root_rot_b,
                root_pos=tar_root_pos_b,
                root_rot=tar_root_rot_b,
                joint_rot=tar_joint_rot_b,
                key_pos=tar_key_pos_b,
                global_obs=self._global_obs,
                root_height_obs=self._root_height_obs,
            )
            tar_obs_b = torch.reshape(tar_obs_b, [tar_obs_b.shape[0], tar_obs_b.shape[1] * tar_obs_b.shape[2]])
            obs_b = torch.cat([obs_b, tar_obs_b], dim=-1)

        rel_root = root_pos_b - root_pos_a
        if not self._global_obs:
            heading = torch_util.calc_heading_quat_inv(root_rot_a)
            rel_root = torch_util.quat_rotate(heading, rel_root)

        obs = [obs_a, obs_b, rel_root]
        if self._enable_phase_obs:
            phase = self._motion_lib_a.calc_motion_phase(motion_ids, motion_times)
            phase_obs = deepmimic_env.compute_phase_obs(phase, self._num_phase_encoding)
            obs.append(phase_obs)
        return torch.cat(obs, dim=-1)

    def _fetch_tar_obs_data(self, motion_lib_obj, motion_ids, motion_times):
        n = motion_ids.shape[0]
        num_steps = self._tar_obs_steps.shape[0]
        assert num_steps > 0

        motion_times = motion_times.unsqueeze(-1)
        time_steps = self._engine.get_timestep() * self._tar_obs_steps
        motion_times = motion_times + time_steps
        motion_ids_tiled = torch.broadcast_to(motion_ids.unsqueeze(-1), motion_times.shape)

        motion_ids_tiled = motion_ids_tiled.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, _, _, joint_rot, _ = motion_lib_obj.calc_motion_frame(motion_ids_tiled, motion_times)

        root_pos = root_pos.reshape([n, num_steps, root_pos.shape[-1]])
        root_rot = root_rot.reshape([n, num_steps, root_rot.shape[-1]])
        joint_rot = joint_rot.reshape([n, num_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
        return root_pos, root_rot, joint_rot

    def _track_global_root(self):
        return self._enable_tar_obs and self._global_obs

    def _apply_action(self, actions):
        clip = torch.minimum(torch.maximum(actions, self._action_bound_low), self._action_bound_high)
        if self._control_mode == "dual":
            assert clip.shape[-1] == self._num_action_dof_a + self._num_action_dof_b, (
                "action width {:d} != dof_a + dof_b ({:d} + {:d})".format(
                    clip.shape[-1], self._num_action_dof_a, self._num_action_dof_b
                )
            )
            act_a = clip[..., : self._num_action_dof_a]
            act_b = clip[..., self._num_action_dof_a :]
        elif self._control_mode == "single_a":
            act_a = clip
            self._engine.set_cmd(self._char_a_ids[0], act_a)
        elif self._control_mode == "single_b":
            act_b = clip
            self._engine.set_cmd(self._char_b_ids[0], act_b)
        if self._control_mode == "dual":
            self._engine.set_cmd(self._char_a_ids[0], act_a)
            self._engine.set_cmd(self._char_b_ids[0], act_b)

    def _update_reward(self):
        root_pos_a = self._s_root_pos_a
        root_rot_a = self._s_root_rot_a
        root_vel_a = self._s_root_vel_a
        root_ang_vel_a = self._s_root_ang_vel_a
        dof_vel_a = self._s_dof_vel_a
        body_pos_a = self._s_body_pos_a
        joint_rot_a = self._s_joint_rot_a
        key_pos_a = body_pos_a[..., self._key_body_ids_a, :] if self._key_body_ids_a.numel() > 0 else torch.zeros([0], device=self._device)
        tar_key_pos_a = self._ref_body_pos_a[..., self._key_body_ids_a, :] if self._key_body_ids_a.numel() > 0 else torch.zeros([0], device=self._device)
        rew_a = deepmimic_env.compute_reward(
            root_pos_a,
            root_rot_a,
            root_vel_a,
            root_ang_vel_a,
            joint_rot_a,
            dof_vel_a,
            key_pos_a,
            self._ref_root_pos_a,
            self._ref_root_rot_a,
            self._ref_root_vel_a,
            self._ref_root_ang_vel_a,
            self._ref_joint_rot_a,
            self._ref_dof_vel_a,
            tar_key_pos_a,
            self._joint_err_w_a,
            self._dof_err_w_a,
            self._track_root_h,
            self._track_root,
            self._reward_pose_w,
            self._reward_vel_w,
            self._reward_root_pose_w,
            self._reward_root_vel_w,
            self._reward_key_pos_w,
            self._reward_pose_scale,
            self._reward_vel_scale,
            self._reward_root_pose_scale,
            self._reward_root_vel_scale,
            self._reward_key_pos_scale,
        )

        root_pos_b = self._s_root_pos_b
        root_rot_b = self._s_root_rot_b
        root_vel_b = self._s_root_vel_b
        root_ang_vel_b = self._s_root_ang_vel_b
        dof_vel_b = self._s_dof_vel_b
        body_pos_b = self._s_body_pos_b
        joint_rot_b = self._s_joint_rot_b
        key_pos_b = body_pos_b[..., self._key_body_ids_b, :] if self._key_body_ids_b.numel() > 0 else torch.zeros([0], device=self._device)
        tar_key_pos_b = self._ref_body_pos_b[..., self._key_body_ids_b, :] if self._key_body_ids_b.numel() > 0 else torch.zeros([0], device=self._device)
        rew_b = deepmimic_env.compute_reward(
            root_pos_b,
            root_rot_b,
            root_vel_b,
            root_ang_vel_b,
            joint_rot_b,
            dof_vel_b,
            key_pos_b,
            self._ref_root_pos_b,
            self._ref_root_rot_b,
            self._ref_root_vel_b,
            self._ref_root_ang_vel_b,
            self._ref_joint_rot_b,
            self._ref_dof_vel_b,
            tar_key_pos_b,
            self._joint_err_w_b,
            self._dof_err_w_b,
            self._track_root_h,
            self._track_root,
            self._reward_pose_w,
            self._reward_vel_w,
            self._reward_root_pose_w,
            self._reward_root_vel_w,
            self._reward_key_pos_w,
            self._reward_pose_scale,
            self._reward_vel_scale,
            self._reward_root_pose_scale,
            self._reward_root_vel_scale,
            self._reward_key_pos_scale,
        )

        if self._key_body_ids_a.numel() > 0 and self._key_body_ids_b.numel() > 0:
            rel_sim = key_pos_a - key_pos_b
            rel_ref = tar_key_pos_a - tar_key_pos_b
            rel_err = torch.mean(torch.sum((rel_ref - rel_sim) ** 2, dim=-1), dim=-1)
            rel_rew = torch.exp(-self._reward_rel_key_pos_scale * rel_err)
        else:
            rel_rew = torch.zeros_like(rew_a)

        self._reward_buf[:] = 0.5 * (rew_a + rew_b) + self._reward_rel_key_pos_w * rel_rew

        if self._reward_contact_w > 0.0 and self._contact_body_ids_inter_a.numel() > 0:
            self._reward_buf[:] += self._reward_contact_w * self._compute_contact_reward()

        if self._reward_balance_w > 0.0 or self._reward_upright_w > 0.0 or self._reward_foot_contact_w > 0.0:
            aux_a = deepmimic_env.compute_auxiliary_reward(
                root_pos=root_pos_a, root_rot=root_rot_a,
                ground_contact_force=self._s_gcf_a, contact_body_ids=self._contact_body_ids_a,
                target_root_height=self._target_root_height,
                balance_w=self._reward_balance_w, upright_w=self._reward_upright_w,
                foot_contact_w=self._reward_foot_contact_w,
            )
            aux_b = deepmimic_env.compute_auxiliary_reward(
                root_pos=root_pos_b, root_rot=root_rot_b,
                ground_contact_force=self._s_gcf_b, contact_body_ids=self._contact_body_ids_b,
                target_root_height=self._target_root_height,
                balance_w=self._reward_balance_w, upright_w=self._reward_upright_w,
                foot_contact_w=self._reward_foot_contact_w,
            )
            self._reward_buf[:] += 0.5 * (aux_a + aux_b)

    def _compute_contact_reward(self):
        """Dense reward for actor A's contact bodies approaching actor B's contact bodies.

        Returns a scalar per env in [0, 1] based on minimum pairwise distance
        between the two body sets, shaped as exp(-scale * min_dist).
        """
        body_pos_a = self._s_body_pos_a  # [n, n_bodies, 3]
        body_pos_b = self._s_body_pos_b

        pos_a = body_pos_a[..., self._contact_body_ids_inter_a, :]  # [n, n_ca, 3]
        pos_b = body_pos_b[..., self._contact_body_ids_inter_b, :]  # [n, n_cb, 3]

        # Pairwise squared distances [n, n_ca, n_cb]
        diff = pos_a.unsqueeze(-2) - pos_b.unsqueeze(-3)
        sq_dists = (diff * diff).sum(dim=-1)
        min_sq_dist = sq_dists.view(sq_dists.shape[0], -1).min(dim=-1).values
        min_dist = torch.sqrt(min_sq_dist.clamp(min=0.0))

        return torch.exp(-self._reward_contact_scale * min_dist)

    def _update_done(self):
        times = self._get_motion_times()
        motion_len_a = self._motion_lib_a.get_motion_length(self._motion_ids)
        motion_len_b = self._motion_lib_b.get_motion_length(self._motion_ids)
        motion_len = torch.minimum(motion_len_a, motion_len_b)
        motion_len_term_a = self._motion_lib_a.get_motion_loop_mode(self._motion_ids) != motion.LoopMode.WRAP.value
        motion_len_term_b = self._motion_lib_b.get_motion_loop_mode(self._motion_ids) != motion.LoopMode.WRAP.value
        motion_len_term = torch.logical_and(motion_len_term_a, motion_len_term_b)
        track_root = self._track_global_root()

        done_a = deepmimic_env.compute_done(
            done_buf=self._done_buf,
            time=self._time_buf,
            ep_len=self._episode_length,
            root_rot=self._s_root_rot_a,
            body_pos=self._s_body_pos_a,
            tar_root_rot=self._ref_root_rot_a,
            tar_body_pos=self._ref_body_pos_a,
            ground_contact_force=self._s_gcf_a,
            contact_body_ids=self._contact_body_ids_a,
            pose_termination=self._pose_termination,
            pose_termination_dist=self._pose_termination_dist,
            global_obs=self._global_obs,
            enable_early_termination=self._enable_early_termination,
            motion_times=times,
            motion_len=motion_len,
            motion_len_term=motion_len_term,
            track_root=track_root,
            pose_termination_body_ids=self._pose_termination_body_ids_a,
        )

        done_b = deepmimic_env.compute_done(
            done_buf=self._done_buf,
            time=self._time_buf,
            ep_len=self._episode_length,
            root_rot=self._s_root_rot_b,
            body_pos=self._s_body_pos_b,
            tar_root_rot=self._ref_root_rot_b,
            tar_body_pos=self._ref_body_pos_b,
            ground_contact_force=self._s_gcf_b,
            contact_body_ids=self._contact_body_ids_b,
            pose_termination=self._pose_termination,
            pose_termination_dist=self._pose_termination_dist,
            global_obs=self._global_obs,
            enable_early_termination=self._enable_early_termination,
            motion_times=times,
            motion_len=motion_len,
            motion_len_term=motion_len_term,
            track_root=track_root,
            pose_termination_body_ids=self._pose_termination_body_ids_b,
        )

        done = torch.full_like(self._done_buf, base_env.DoneFlags.NULL.value)
        done[torch.logical_or(done_a == base_env.DoneFlags.TIME.value, done_b == base_env.DoneFlags.TIME.value)] = base_env.DoneFlags.TIME.value
        done[torch.logical_and(done_a == base_env.DoneFlags.SUCC.value, done_b == base_env.DoneFlags.SUCC.value)] = base_env.DoneFlags.SUCC.value
        done[torch.logical_or(done_a == base_env.DoneFlags.FAIL.value, done_b == base_env.DoneFlags.FAIL.value)] = base_env.DoneFlags.FAIL.value
        self._done_buf[:] = done

    def _print_char_prop(self, env_id, obj_id, name):
        num_dofs = self._engine.get_obj_num_dofs(obj_id)
        total_mass = self._engine.calc_obj_mass(env_id, obj_id)
        Logger.print("Char {} {}\n\tDoFs: {}\n\tMass: {:.3f} kg\n".format(name, obj_id, num_dofs, total_mass))

    def _validate_envs(self):
        sim_a = self._engine.get_obj_body_names(self._char_a_ids[0])
        sim_b = self._engine.get_obj_body_names(self._char_b_ids[0])
        kin_a = self._kin_char_model_a.get_body_names()
        kin_b = self._kin_char_model_b.get_body_names()
        for s, k in zip(sim_a, kin_a):
            assert s == k
        for s, k in zip(sim_b, kin_b):
            assert s == k
