"""Roll out an env with uniform random actions in the Box bounds (sanity check for control).

Single SMPL (`data/assets/smpl/smpl.xml`, default):
  python tools/random_action_rollout.py \\
    --engine_config data/engines/newton_engine.yaml \\
    --num_envs 4 --steps 120

Dual G1 + fat:
  python tools/random_action_rollout.py \\
    --env_config data/envs/dual_deepmimic_g1_fat_env.yaml

Dual SMPL + viewer:
  python tools/random_action_rollout.py \\
    --env_config data/envs/dual_deepmimic_smpl_upper_only_env.yaml \\
    --visualize --num_envs 1

With `--visualize`, frame pacing defaults to ~30 FPS unless you set `--sleep 0`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMICKIT_ROOT = os.path.join(REPO_ROOT, "mimickit")
sys.path.insert(0, MIMICKIT_ROOT)
sys.path.insert(0, REPO_ROOT)

import envs.env_builder as env_builder


def main():
    parser = argparse.ArgumentParser(description="Random-action rollout for control sanity check.")
    parser.add_argument(
        "--env_config",
        default="data/envs/deepmimic_smpl_env.yaml",
        help="Use deepmimic_smpl_env.yaml for single smpl.xml (default).",
    )
    parser.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default=None, help="cuda:0 or cpu (default: cuda if available)")
    parser.add_argument("--visualize", action="store_true", help="Open Newton viewer (pyglet); pair with --num_envs 1 for a single scene.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Seconds to sleep after each step when visualizing. Default: 1/30 s with --visualize, else 0.",
    )
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    sleep_s = args.sleep
    if sleep_s is None:
        sleep_s = (1.0 / 30.0) if args.visualize else 0.0

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.chdir(REPO_ROOT)

    env = env_builder.build_env(
        args.env_config,
        args.engine_config,
        args.num_envs,
        device,
        visualize=args.visualize,
        record_video=False,
    )

    a_space = env.get_action_space()
    low = torch.tensor(a_space.low, device=device, dtype=torch.float32)
    high = torch.tensor(a_space.high, device=device, dtype=torch.float32)
    width = int(np.prod(a_space.shape))
    print(f"device={device} action_dim={width} num_envs={args.num_envs} steps={args.steps}")
    print(f"action low[:8]={low[:8].cpu().numpy()} high[:8]={high[:8].cpu().numpy()}")

    env.reset()
    if args.visualize:
        # Show reference pose before the first random step.
        env._render()

    env_name = None
    try:
        cfg_path = args.env_config
        import yaml

        with open(cfg_path, "r") as f:
            env_name = yaml.safe_load(f).get("env_name")
    except Exception:
        pass

    char_a_id, char_b_id = 0, None
    if env_name == "dual_deepmimic" and hasattr(env, "_char_a_ids"):
        char_a_id = env._char_a_ids[0]
        if hasattr(env, "_char_b_ids"):
            char_b_id = env._char_b_ids[0]
    elif env_name == "deepmimic" and hasattr(env, "_get_char_id"):
        char_a_id = env._get_char_id()

    for t in range(args.steps):
        u = torch.rand(args.num_envs, width, device=device, dtype=torch.float32)
        actions = low + (high - low) * u
        obs, rew, done, _info = env.step(actions)

        if done.any():
            env.reset()

        if t % args.log_every == 0 or t == args.steps - 1:
            za = env._engine.get_root_pos(char_a_id)[:, 2].mean().item()
            line = (
                f"step {t:4d}  mean_rew={rew.mean().item():.4f}  "
                f"root_z_a={za:.3f}  done_any={done.any().item()}"
            )
            if char_b_id is not None:
                zb = env._engine.get_root_pos(char_b_id)[:, 2].mean().item()
                line += f"  root_z_b={zb:.3f}"
            print(line)

        if args.visualize and sleep_s > 0.0:
            time.sleep(sleep_s)
            try:
                import pyglet

                pyglet.app.platform_event_loop.step(0.0)
            except Exception:
                pass

    if args.visualize:
        print("Rollout finished. Close the viewer window (or press Enter here to exit the script).")
        try:
            input()
        except EOFError:
            pass


if __name__ == "__main__":
    main()
