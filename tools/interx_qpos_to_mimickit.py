import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMICKIT_ROOT = os.path.join(REPO_ROOT, "mimickit")
sys.path.insert(0, MIMICKIT_ROOT)
sys.path.insert(0, REPO_ROOT)

from mimickit.anim.motion import LoopMode, Motion
from mimickit.util.torch_util import quat_mul, quat_rotate, quat_to_exp_map


# InterX paired exports arrive with the root orientation expressed in a basis
# where the character's local up axis is effectively Y-up, while MimicKit's
# SMPL asset expects the root in its Z-up basis. Apply the same fixed basis
# change used by the SMPL converter before writing the motion.
INTERX_ROOT_BASIS_CORRECTION = torch.tensor([-0.5, -0.5, -0.5, 0.5], dtype=torch.float32)
INTERX_UPSIDE_DOWN_CORRECTION = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def _canonicalize_root_quat(root_quat: torch.Tensor) -> torch.Tensor:
    root_quat = quat_mul(
        root_quat,
        INTERX_ROOT_BASIS_CORRECTION.expand(root_quat.shape[0], -1),
    )

    world_up = torch.zeros((root_quat.shape[0], 3), dtype=root_quat.dtype, device=root_quat.device)
    world_up[:, 2] = 1.0
    rotated_up = quat_rotate(root_quat, world_up)

    upside_down = rotated_up[:, 2] < 0.0
    if torch.any(upside_down):
        corrected = quat_mul(
            root_quat[upside_down],
            INTERX_UPSIDE_DOWN_CORRECTION.expand(upside_down.sum(), -1),
        )
        root_quat = root_quat.clone()
        root_quat[upside_down] = corrected

    return root_quat


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _parse_loop_mode(loop_mode: str) -> LoopMode:
    if loop_mode == "wrap":
        return LoopMode.WRAP
    if loop_mode == "clamp":
        return LoopMode.CLAMP
    raise ValueError(f"Unsupported loop mode: {loop_mode}")


def qpos_to_motion(qpos: np.ndarray, fps: int, loop_mode: LoopMode) -> Motion:
    root_pos = qpos[:, :3]
    root_quat = torch.tensor(qpos[:, 3:7], dtype=torch.float32)
    root_quat = _canonicalize_root_quat(root_quat)
    dof_pos = qpos[:, 7:]
    root_expmap = quat_to_exp_map(root_quat).cpu().numpy()
    frames = np.concatenate([root_pos, root_expmap, dof_pos], axis=-1).astype(np.float32)
    return Motion(loop_mode=loop_mode, fps=int(fps), frames=frames)


def _filter_actuator(actuator_node: ET.Element, prefix: str) -> ET.Element:
    out = ET.Element("actuator")
    for child in actuator_node:
        child_joint = child.attrib.get("joint", "")
        child_name = child.attrib.get("name", "")
        if child_joint.startswith(prefix) or child_name.startswith(prefix):
            out.append(copy.deepcopy(child))
    return out


def extract_single_char_mjcf(scene_xml: str, root_body_name: str, output_xml: str) -> None:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No worldbody found in {scene_xml}")

    selected_body = None
    for body in worldbody.findall("body"):
        if body.attrib.get("name") == root_body_name:
            selected_body = copy.deepcopy(body)
            break

    if selected_body is None:
        raise ValueError(f"Could not find body {root_body_name} in {scene_xml}")

    out_root = ET.Element(root.tag, root.attrib)
    for child in root:
        if child.tag in ("worldbody", "contact", "actuator"):
            continue
        out_root.append(copy.deepcopy(child))

    out_worldbody = ET.SubElement(out_root, "worldbody")
    out_worldbody.append(selected_body)

    actuator = root.find("actuator")
    if actuator is not None and len(actuator) > 0:
        prefix = root_body_name.split("Pelvis")[0]
        filtered_actuator = _filter_actuator(actuator, prefix)
        if len(filtered_actuator) > 0:
            out_root.append(filtered_actuator)

    _ensure_dir(output_xml)
    ET.ElementTree(out_root).write(output_xml, encoding="utf-8", xml_declaration=False)


def _infer_scene_xml(input_file: str, data: np.lib.npyio.NpzFile, scene_xml: str | None) -> str:
    if scene_xml is not None:
        return scene_xml

    if "dual_scene_xml" in data.files:
        candidate = str(data["dual_scene_xml"])
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        local_candidate = os.path.join(os.path.dirname(input_file), os.path.basename(candidate))
        if os.path.exists(local_candidate):
            return local_candidate

    default_candidate = os.path.join(
        os.path.dirname(input_file),
        os.path.splitext(os.path.basename(input_file))[0] + "_dual_scene.xml",
    )
    if os.path.exists(default_candidate):
        return default_candidate

    raise ValueError("Could not infer dual scene XML. Provide --scene_xml.")


def convert_dual_bundle(
    input_file: str,
    output_motion_a: str,
    output_motion_b: str,
    output_char_a: str,
    output_char_b: str,
    scene_xml: str | None,
    loop_mode: str,
) -> None:
    data = np.load(input_file, allow_pickle=True)
    required = ["qpos_A", "qpos_B", "fps"]
    for key in required:
        if key not in data.files:
            raise KeyError(f"Missing required key {key} in {input_file}")

    scene_xml = _infer_scene_xml(input_file, data, scene_xml)
    prefix_a = str(data["dual_prefix_A"]) if "dual_prefix_A" in data.files else "A_"
    prefix_b = str(data["dual_prefix_B"]) if "dual_prefix_B" in data.files else "B_"

    motion_loop = _parse_loop_mode(loop_mode)
    fps = int(data["fps"])

    motion_a = qpos_to_motion(data["qpos_A"].astype(np.float32), fps=fps, loop_mode=motion_loop)
    motion_b = qpos_to_motion(data["qpos_B"].astype(np.float32), fps=fps, loop_mode=motion_loop)

    extract_single_char_mjcf(scene_xml, f"{prefix_a}Pelvis", output_char_a)
    extract_single_char_mjcf(scene_xml, f"{prefix_b}Pelvis", output_char_b)

    _ensure_dir(output_motion_a)
    _ensure_dir(output_motion_b)
    motion_a.save(output_motion_a)
    motion_b.save(output_motion_b)

    print(f"Saved motion A to {output_motion_a}")
    print(f"Saved motion B to {output_motion_b}")
    print(f"Saved char A asset to {output_char_a}")
    print(f"Saved char B asset to {output_char_b}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert interx_h2h qpos bundles into MimicKit Motion files and matching MJCF assets."
    )
    parser.add_argument("--input_file", required=True, help="Path to the interx_h2h dual npz bundle")
    parser.add_argument("--output_motion_a", required=True, help="Output path for character A motion pkl")
    parser.add_argument("--output_motion_b", required=True, help="Output path for character B motion pkl")
    parser.add_argument("--output_char_a", required=True, help="Output path for character A MJCF xml")
    parser.add_argument("--output_char_b", required=True, help="Output path for character B MJCF xml")
    parser.add_argument("--scene_xml", default=None, help="Optional override for the dual scene xml")
    parser.add_argument("--loop", default="clamp", choices=["wrap", "clamp"], help="Loop mode")
    args = parser.parse_args()

    convert_dual_bundle(
        input_file=args.input_file,
        output_motion_a=args.output_motion_a,
        output_motion_b=args.output_motion_b,
        output_char_a=args.output_char_a,
        output_char_b=args.output_char_b,
        scene_xml=args.scene_xml,
        loop_mode=args.loop,
    )


if __name__ == "__main__":
    main()
