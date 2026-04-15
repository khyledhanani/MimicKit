# MimicKit

[General framework README](docs/README_MimicKit.md) — installation, all imitation methods (DeepMimic, AMP, …), motion data format, and citation.

## Dual Motion Pipeline

This guide documents the full dual-motion workflow in this repository:

1. Start from Holosoma or EgoHuman motion exports.
2. Convert them into MimicKit `.pkl` motions.
3. Validate and preview the paired motions.
4. Train a dual imitation policy.
5. Test the trained policy and view the reference motions.

All commands below assume you are running from the repository root.

---

## 1. Setup

for the base install:

- install a supported simulator backend, Newton for the dual setups in this repo:

<summary>Newton</summary>

Install [Newton](https://newton-physics.github.io/newton/guide/installation.html). This framework has been tested with `v1.0.0`.

To use Newton, specify the argument `--engine_config data/engines/newton_engine.yaml` when running the code (can generally specify this in the args files).
- `pip install -r requirements.txt`
- download data from: https://drive.google.com/drive/folders/1g-PhQgRvmTFn-bhEDrxLhECDYDQ8s9MI?usp=sharing

The main setup 

For the dual examples already configured here, the most important runtime files are:

- `mimickit/run.py`: main train/test entrypoint
- `mimickit/envs/dual_deepmimic_env.py`: dual imitation environment
- `mimickit/envs/view_motion_dual_env.py`: paired motion viewer environment
- `tools/interx_qpos_to_mimickit.py`: converts retargeted dual `qpos_A` / `qpos_B` bundles to MimicKit motions
- `tools/smpl_to_mimickit/interx_smpl_to_mimickit.py`: converts Y-up SMPL archives to MimicKit
- `tools/smpl_to_mimickit/mpc_joints_to_mimickit.py`: converts Z-up `mpc_joints` SMPL archives to MimicKit
- `tools/validate_mimickit_motion.py`: checks a converted `.pkl` against a target rig (ensuring that the motion matches the body we want to mimic)

---

## 2. Which Converter To Use

Files from the retargeting process or optimizers need to be converted before they are easily used by the mimicking infra

Use the converter that matches the source archive schema, not the dataset name.

| Source export | Typical keys | Use this converter |
| --- | --- | --- |
| Raw InterX / AMASS / standard SMPL | `poses` + `trans`, or `root_orient` / `global_orient` + `pose_body` + `trans` | `tools/smpl_to_mimickit/interx_smpl_to_mimickit.py` |
| EgoHuman `mpc_joints` paired files | `root_orient`, `global_orient`, `trans`, `pose_body` | `tools/smpl_to_mimickit/mpc_joints_to_mimickit.py` |
| Holosoma or retargeted dual bundle | `qpos_A`, `qpos_B`, usually plus `fps`, `dual_scene_xml`, `dual_prefix_A`, `dual_prefix_B` | `tools/interx_qpos_to_mimickit.py` |

Examples already present in this repo:

- Raw InterX SMPL-X clips:
  - `data/motions/dual/original/G001T003A016R008/P1.npz`
  - `data/motions/dual/original/G001T003A016R008/P2.npz`
- EgoHuman `mpc_joints` paired clips:
  - `data/motions/dual/original/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric/P1.npz`
  - `data/motions/dual/original/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric/P2.npz`
- EgoHuman or Holosoma-style dual retarget bundles:
  - `data/motions/dual/original/egohuman_dual_g1_upper_back_symmetric_no_y2z/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric.npz`
  - `data/motions/dual/original/egohuman_dual_g1_fat_sharedscale_20260410_174839/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric.npz`

---

## 3. Source Pipelines

### A. Raw InterX SMPL-X to MimicKit without retargeting

If you only want to view or train on paired human SMPL motions directly, convert `P1.npz` and `P2.npz` separately:

```bash
python tools/smpl_to_mimickit/interx_smpl_to_mimickit.py --input_file dual_data/G001T003A016R008/P1.npz --output_file dual_data/converted/G001T003A016R008_p1.pkl --loop clamp --z_correction full --input_fps 30

python tools/smpl_to_mimickit/interx_smpl_to_mimickit.py --input_file dual_data/G001T003A016R008/P2.npz --output_file dual_data/converted/G001T003A016R008_p2.pkl --loop clamp --z_correction full --input_fps 30
```

Those files are already wired into `data/envs/dual_deepmimic_smpl_g001t003a016r008_env.yaml`.

### C. EgoHuman `mpc_joints` paired SMPL exports to MimicKit

The paired EgoHuman source files under `dual_data/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric/` contain Z-up SMPL motion with keys such as:

- `root_orient`
- `global_orient`
- `trans`
- `pose_body`

Convert them with the `mpc_joints` converter:

```bash
python tools/smpl_to_mimickit/mpc_joints_to_mimickit.py --input_file dual_data/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric/P1.npz --output_file dual_data/converted/p1_upper_back_symmetric.pkl --loop clamp --z_correction full --input_fps 30

python tools/smpl_to_mimickit/mpc_joints_to_mimickit.py --input_file dual_data/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric/P2.npz --output_file dual_data/converted/p2_upper_back_symmetric.pkl --loop clamp --z_correction full --input_fps 30
```

These are the paired human clips used by `data/envs/view_motion_dual_smpl_env.yaml`.

### D. Holosoma retargeted dual bundles to MimicKit

When the source has already been retargeted and saved as one dual bundle with `qpos_A` and `qpos_B`, convert both sides in one step:

```bash
python tools/interx_qpos_to_mimickit.py --input_file dual_data/egohuman_dual_g1_upper_back_symmetric_no_y2z/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric.npz --output_motion_a dual_data/converted/egohuman_dual_g1_upper_back_symmetric_g1_a.pkl --output_motion_b dual_data/converted/egohuman_dual_g1_upper_back_symmetric_g1_b.pkl --char_file_a data/assets/g1/g1.xml --char_file_b data/assets/g1/g1.xml --loop clamp --input_fps 30
```

For the mixed G1 + human-variant setup, the exact same converter is used, but with different target rigs:

```bash
python tools/interx_qpos_to_mimickit.py --input_file dual_data/egohuman_dual_g1_fat_sharedscale_20260410_174839/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric.npz --output_motion_a dual_data/converted/egohuman_g1_fat_g1.pkl --output_motion_b dual_data/converted/egohuman_g1_fat_fat.pkl --char_file_a data/assets/g1/g1.xml --char_file_b data/assets/smpl/human_variants/fat.xml --loop clamp --input_fps 30
```

Use this converter whenever the archive already has robot or articulated-character `qpos` data. Use the SMPL converters only for raw `poses` / `trans` style source files.

---

## 4. Validate And Preview Converted Motions

Validate each converted motion before training:

```bash
python tools/validate_mimickit_motion.py --motion_file dual_data/converted/egohuman_dual_g1_upper_back_symmetric_g1_a.pkl --char_file data/assets/g1/g1.xml

python tools/validate_mimickit_motion.py --motion_file dual_data/converted/egohuman_g1_fat_fat.pkl --char_file data/assets/smpl/human_variants/fat.xml
```

Two good viewing options:

1. Simple paired viewer for the raw converted clips:

```bash
python tools/view_paired_motion.py --char_file_a data/assets/g1/g1.xml --char_file_b data/assets/g1/g1.xml --motion_file_a dual_data/converted/egohuman_dual_g1_upper_back_symmetric_g1_a.pkl --motion_file_b dual_data/converted/egohuman_dual_g1_upper_back_symmetric_g1_b.pkl
```

2. MimicKit viewer env using the repo arg presets:

```bash
python mimickit/run.py --arg_file args/view_motion_dual_g1_upper_back_symmetric_args.txt --visualize true
python mimickit/run.py --arg_file args/view_motion_dual_g1_fat_args.txt --visualize true
python mimickit/run.py --arg_file args/view_motion_dual_smpl_args.txt --visualize true
```

If the motion looks rotated or upside-down after conversion, the repair helpers are:

- `tools/fix_mimickit_motion_basis.py`
- `tools/repair_mimickit_motion.py`

---

## 5. Wire The Motion Files Into A Dual Env

The dual environment YAML is where the converted motions become training data. The key fields are:

- `char_file_a` / `char_file_b`: the actor rigs
- `motion_file_a` / `motion_file_b`: the converted MimicKit clips
- `init_pose_a` / `init_pose_b` or `init_pose`
- reward weights, contact bodies, key bodies, and control mode

The main dual env presets already in this repo are:

For the unitree g1 hug from the optimizer (both are g1)
- `data/envs/dual_deepmimic_g1_upper_back_symmetric_env.yaml`
- `data/envs/view_motion_dual_g1_upper_back_symmetric_env.yaml`

For the smpl humanoids hug from the optimizer (both are smpl type bodies)
- `data/envs/dual_deepmimic_smpl_upper_only_env.yaml.yaml`
- `data/envs/view_motion_dual_smpl_env.yaml`

The corresponding arg presets are:

- `args/dual_deepmimic_g1_upper_back_symmetric_newton_ppo_args.txt`
- `args/view_motion_dual_g1_upper_back_symmetric_args.txt`
- `args/dual_deepmimic_smpl_upper_only_newton_ppo_args.txt`
- `args/view_motion_dual_smpl_args.txt`

---

## 6. Train


Depending on the run, you can add in --num_envs to specify the number of parralel environments you want to train. Generally, a 12GB VRAM device can fit 2048 - 4096 parralel envs depending on the setup.


### Dual G1 upper-back symmetric

```bash
python mimickit/run.py --arg_file args/dual_deepmimic_g1_upper_back_symmetric_newton_ppo_args.txt --visualize false
```

### Dual SMPL human baseline


```bash
python mimickit/run.py --arg_file args/dual_deepmimic_smpl_upper_back_newton_ppo_args.txt --visualize true
```

Training outputs are written to the `out_dir` from the arg file, for example:

- `output/dual_deepmimic_g1_upper_back_symmetric/`
- `output/dual_deepmimic_g1_fat/`
- `output/dual_deepmimic_smpl_upper_back/`

Each run saves:

- `model.pt`: latest policy checkpoint
- `log.txt`: text training log
- copied config snapshots: `engine_config.yaml`, `env_config.yaml`, `agent_config.yaml`
- optionally `int_models/` if intermediate checkpoints are enabled

---

## 7. Test A Trained Policy

Testing uses the same arg file plus `--mode test` and `--model_file`.

Example for the G1 upper-back symmetric setup:

```bash
python mimickit/run.py --arg_file args/dual_deepmimic_g1_upper_back_symmetric_newton_ppo_args.txt --mode test --num_envs 1 --visualize true --model_file output/dual_deepmimic_g1_upper_back_symmetric/model.pt
```

Example for the G1 + fat setup:

```bash
python mimickit/run.py --arg_file args/dual_deepmimic_g1_fat_newton_ppo_args.txt --mode test --num_envs 1 --visualize true --model_file output/dual_deepmimic_g1_fat/model.pt
```

---

## 8. Quick File Map

- `dual_data/`: example source files, retarget bundles, and converted dual motions
- `dual_data/converted/`: converted MimicKit `.pkl` motion files used by the env presets
- `holosoma/`: retargeting pipeline used to produce dual `qpos_A` / `qpos_B` bundles
- `egohuman-rl/`: source single-human motion data and related physics backend integration
- `data/envs/`: dual training and view environment presets
- `data/agents/dual_deepmimic_g1_fat_ppo_agent.yaml`: PPO config used by the dual G1 presets
- `args/`: runnable command presets for training and viewing
- `output/`: checkpoints and logs after training

---

## 9. Practical Rule Of Thumb

- If the file has `qpos_A` and `qpos_B`, use `tools/interx_qpos_to_mimickit.py`.
- If the file has `P1.npz` / `P2.npz` with SMPL pose arrays and `trans`, use one of the SMPL converters.
- Validate each `.pkl` against the exact target rig before starting a long training run.
- Once the converted motion paths in `data/envs/*.yaml` are correct, training, testing, and viewing all go through `mimickit/run.py`.
