# SMPL to MimicKit Motion Converters

Tools convert SMPL-parameterized `.npz` motion into MimicKit `.pkl` format for `data/assets/smpl/smpl.xml`.

Run all commands from the **repository root**.

## Which script?

| Script | Input format |
|--------|----------------|
| `interx_smpl_to_mimickit.py` | **InterX / AMASS / standard SMPL Y-up**: `trans` has **Y = height**; `root_orient` in SMPL Y-up. |
| `mpc_joints_to_mimickit.py` | **mpc_joints exports**: `trans` already **Z-up** (Z = height); `root_orient` uses the basis correction (same 120° quaternion as `interx_qpos_to_mimickit`). |
| `smpl_to_mimickit.py` | **Legacy** wrapper; supports deprecated `--coord_system zup`. Prefer the two scripts above. |

MuJoCo **qpos** bundles (`qpos_A` / `qpos_B`) use `tools/interx_qpos_to_mimickit.py` instead.

## Shared arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_file` | required | Input `.npz` |
| `--output_file` | required | Output `.pkl` |
| `--loop` | `wrap` | `wrap` or `clamp` |
| `--start_frame` | `0` | Clip start |
| `--end_frame` | `-1` | Clip end (`-1` = all) |
| `--output_fps` | `-1` | Output FPS (`-1` = source) |
| `--input_fps` | `-1` | Use if FPS missing from archive |
| `--z_correction` | `calibrate` | `none`, `calibrate` (first 30 frames), or `full` (entire clip) |

## Examples

InterX / Y-up (e.g. `data/motions/dual/original/G001T003A016R008/P1.npz`):

```bash
python tools/smpl_to_mimickit/interx_smpl_to_mimickit.py \
    --input_file data/motions/dual/original/G001T003A016R008/P1.npz \
    --output_file data/motions/dual/converted/G001T003A016R008_p1.pkl \
    --loop clamp --z_correction full --input_fps 30
```

mpc_joints / Z-up:

```bash
python tools/smpl_to_mimickit/mpc_joints_to_mimickit.py \
    --input_file data/motions/dual/original/mpc_joints_c_interact_environment-simple_hug_scenario-upper_back_symmetric/P1.npz \
    --output_file data/motions/dual/converted/p1_upper_back_symmetric.pkl \
    --loop clamp --z_correction full --input_fps 30
```

AMASS-style (`poses` + `trans`):

```bash
python tools/smpl_to_mimickit/interx_smpl_to_mimickit.py \
    --input_file data/amass_sample/walk.npz \
    --output_file motions/smpl_walk.pkl \
    --z_correction full --loop wrap
```

## Input schema

Either:

- `poses` + `trans` + `mocap_framerate` or `fps`, or  
- `root_orient` or `global_orient` + `pose_body` + `trans` + optional `fps`

## Credits

Part of the conversion logic is adapted from [PHC](https://github.com/ZhengyiLuo/PHC).
