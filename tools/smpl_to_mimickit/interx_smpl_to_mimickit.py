"""
Convert InterX / AMASS / standard SMPL Y-up .npz exports to MimicKit motion (.pkl).

Run from the MimicKit repository root:

    python tools/smpl_to_mimickit/interx_smpl_to_mimickit.py \\
        --input_file dual_data/G001T003A016R008/P1.npz \\
        --output_file dual_data/converted/out.pkl \\
        --loop clamp --z_correction full --input_fps 30

For mpc_joints (Z-up trans) exports, use mpc_joints_to_mimickit.py instead.
"""

import argparse
import sys

sys.path.append(".")

from tools.smpl_to_mimickit.smpl_mimickit_common import convert_interx_smpl_to_mimickit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert InterX / AMASS / SMPL Y-up motion (.npz) to MimicKit format."
    )
    parser.add_argument("--input_file", required=True, help="Input .npz (Y-up trans, SMPL root_orient)")
    parser.add_argument("--output_file", required=True, help="Output .pkl")
    parser.add_argument("--loop", default="wrap", choices=["wrap", "clamp"], help="Loop mode (default: wrap)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame (default: 0)")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame (-1 = all)")
    parser.add_argument("--output_fps", type=int, default=-1, help="Output FPS (-1 = source)")
    parser.add_argument("--input_fps", type=int, default=-1, help="Override FPS if missing from npz")
    parser.add_argument(
        "--z_correction",
        type=str,
        default="calibrate",
        choices=["none", "calibrate", "full"],
        help="Ground height correction (default: calibrate)",
    )
    args = parser.parse_args()

    convert_interx_smpl_to_mimickit(
        args.input_file,
        args.output_file,
        loop_mode=args.loop,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        output_fps=args.output_fps,
        z_correction=args.z_correction,
        input_fps=args.input_fps,
    )


if __name__ == "__main__":
    main()
