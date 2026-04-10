import argparse
import copy
import os
import pickle
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


def _build_char_model(char_file: str, device: str = "cpu"):
    _, file_ext = os.path.splitext(char_file)
    if file_ext == ".xml":
        import mimickit.anim.mjcf_char_model as mjcf_char_model

        model = mjcf_char_model.MJCFCharModel(device)
    elif file_ext == ".urdf":
        import mimickit.anim.urdf_char_model as urdf_char_model

        model = urdf_char_model.URDFCharModel(device)
    elif file_ext == ".usd":
        import mimickit.anim.usd_char_model as usd_char_model

        model = usd_char_model.USDCharModel(device)
    else:
        raise ValueError(f"Unsupported character file format: {file_ext}")

    model.load(char_file)
    return model


def _load_bundle(input_file: str):
    _, ext = os.path.splitext(input_file)
    if ext == ".npz":
        return np.load(input_file, allow_pickle=True)
    if ext in (".pkl", ".pickle"):
        with open(input_file, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported bundle format: {ext}")


def _bundle_has(data, key: str) -> bool:
    if hasattr(data, "files"):
        return key in data.files
    return key in data


def _bundle_get(data, key: str):
    if hasattr(data, "files"):
        return data[key]
    return data[key]


def _bundle_get_first(data, keys: list[str], field_name: str):
    for key in keys:
        if _bundle_has(data, key):
            return _bundle_get(data, key), key
    raise KeyError(f"Could not find {field_name}. Tried keys: {keys}")


def _to_str(value) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except ValueError:
            pass
    return str(value)


def _reorder_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return quat[..., [1, 2, 3, 0]]


def _resolve_actor_pose(
    data,
    *,
    side: str,
    qpos_key: str | None,
    root_pos_key: str | None,
    root_rot_key: str | None,
    dof_pos_key: str | None,
) -> tuple[np.ndarray, str]:
    qpos_candidates = [qpos_key] if qpos_key is not None else [
        f"qpos_{side}",
        f"qpos_{side.upper()}",
        f"qpos_{side.lower()}",
    ]
    for key in qpos_candidates:
        if key is not None and _bundle_has(data, key):
            qpos = np.asarray(_bundle_get(data, key), dtype=np.float32)
            return qpos, f"qpos:{key}"

    root_pos_candidates = [root_pos_key] if root_pos_key is not None else [
        f"root_pos_{side}",
        f"root_pos_{side.upper()}",
        f"root_pos_{side.lower()}",
    ]
    root_rot_candidates = [root_rot_key] if root_rot_key is not None else [
        f"root_rot_{side}",
        f"root_rot_{side.upper()}",
        f"root_rot_{side.lower()}",
    ]
    dof_pos_candidates = [dof_pos_key] if dof_pos_key is not None else [
        f"dof_pos_{side}",
        f"dof_pos_{side.upper()}",
        f"dof_pos_{side.lower()}",
    ]

    root_pos, root_pos_src = _bundle_get_first(data, root_pos_candidates, f"root_pos for side {side}")
    root_rot, root_rot_src = _bundle_get_first(data, root_rot_candidates, f"root_rot for side {side}")
    dof_pos, dof_pos_src = _bundle_get_first(data, dof_pos_candidates, f"dof_pos for side {side}")

    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot = np.asarray(root_rot, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=-1)
    return qpos, f"split:{root_pos_src},{root_rot_src},{dof_pos_src}"


def _validate_qpos_shape(qpos: np.ndarray, char_file: str, side: str) -> None:
    char_model = _build_char_model(char_file)
    expected = 7 + char_model.get_dof_size()
    if qpos.ndim != 2:
        raise ValueError(f"{side}: expected qpos to be rank-2, got shape {qpos.shape}")
    if qpos.shape[1] != expected:
        raise ValueError(
            f"{side}: qpos width {qpos.shape[1]} does not match {char_file} "
            f"(expected 7 + dof_size = {expected})"
        )


def qpos_to_motion(
    qpos: np.ndarray,
    fps: int,
    loop_mode: LoopMode,
    *,
    quat_format: str,
    root_basis_correction: str,
) -> Motion:
    root_pos = qpos[:, :3]
    root_quat_np = qpos[:, 3:7]
    if quat_format == "wxyz":
        root_quat_np = _reorder_wxyz_to_xyzw(root_quat_np)
    elif quat_format != "xyzw":
        raise ValueError(f"Unsupported quaternion format: {quat_format}")

    root_quat = torch.tensor(root_quat_np, dtype=torch.float32)
    if root_basis_correction == "interx":
        root_quat = _canonicalize_root_quat(root_quat)
    elif root_basis_correction != "none":
        raise ValueError(f"Unsupported root basis correction: {root_basis_correction}")
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
    output_char_a: str | None,
    output_char_b: str | None,
    scene_xml: str | None,
    loop_mode: str,
    fps_key: str,
    qpos_key_a: str | None,
    qpos_key_b: str | None,
    root_pos_key_a: str | None,
    root_pos_key_b: str | None,
    root_rot_key_a: str | None,
    root_rot_key_b: str | None,
    dof_pos_key_a: str | None,
    dof_pos_key_b: str | None,
    char_file_a: str | None,
    char_file_b: str | None,
    input_fps: int,
    quat_format: str,
    root_basis_correction: str,
) -> None:
    data = _load_bundle(input_file)
    qpos_a, src_a = _resolve_actor_pose(
        data,
        side="A",
        qpos_key=qpos_key_a,
        root_pos_key=root_pos_key_a,
        root_rot_key=root_rot_key_a,
        dof_pos_key=dof_pos_key_a,
    )
    qpos_b, src_b = _resolve_actor_pose(
        data,
        side="B",
        qpos_key=qpos_key_b,
        root_pos_key=root_pos_key_b,
        root_rot_key=root_rot_key_b,
        dof_pos_key=dof_pos_key_b,
    )

    try:
        fps_entry, fps_src = _bundle_get_first(data, [fps_key, "fps", "mocap_framerate"], "fps")
        fps = int(np.asarray(fps_entry).item())
    except KeyError:
        if input_fps <= 0:
            raise
        fps = int(input_fps)
        fps_src = "cli:input_fps"

    if char_file_a is not None:
        _validate_qpos_shape(qpos_a, char_file_a, "A")
    if char_file_b is not None:
        _validate_qpos_shape(qpos_b, char_file_b, "B")

    if output_char_a is not None or output_char_b is not None:
        if output_char_a is None or output_char_b is None:
            raise ValueError("Provide both --output_char_a and --output_char_b, or neither.")
        if not hasattr(data, "files"):
            raise ValueError("Auto-extracting character MJCFs is only supported for .npz bundles.")
        scene_xml = _infer_scene_xml(input_file, data, scene_xml)
        prefix_a = _to_str(data["dual_prefix_A"]) if "dual_prefix_A" in data.files else "A_"
        prefix_b = _to_str(data["dual_prefix_B"]) if "dual_prefix_B" in data.files else "B_"

    motion_loop = _parse_loop_mode(loop_mode)
    motion_a = qpos_to_motion(
        qpos_a.astype(np.float32),
        fps=fps,
        loop_mode=motion_loop,
        quat_format=quat_format,
        root_basis_correction=root_basis_correction,
    )
    motion_b = qpos_to_motion(
        qpos_b.astype(np.float32),
        fps=fps,
        loop_mode=motion_loop,
        quat_format=quat_format,
        root_basis_correction=root_basis_correction,
    )

    if output_char_a is not None:
        extract_single_char_mjcf(scene_xml, f"{prefix_a}Pelvis", output_char_a)
        extract_single_char_mjcf(scene_xml, f"{prefix_b}Pelvis", output_char_b)

    _ensure_dir(output_motion_a)
    _ensure_dir(output_motion_b)
    motion_a.save(output_motion_a)
    motion_b.save(output_motion_b)

    print(f"Saved motion A to {output_motion_a}")
    print(f"Saved motion B to {output_motion_b}")
    print(f"Source A: {src_a}")
    print(f"Source B: {src_b}")
    print(f"FPS source: {fps_src}")
    if char_file_a is not None:
        print(f"Validated A against {char_file_a}")
    if char_file_b is not None:
        print(f"Validated B against {char_file_b}")
    if output_char_a is not None:
        print(f"Saved char A asset to {output_char_a}")
        print(f"Saved char B asset to {output_char_b}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert dual retarget bundles into MimicKit Motion files. "
            "Supports qpos_A/qpos_B bundles and split root_pos/root_rot/dof_pos schemas."
        )
    )
    parser.add_argument("--input_file", required=True, help="Path to the dual retarget bundle (.npz or .pkl)")
    parser.add_argument("--output_motion_a", required=True, help="Output path for character A motion pkl")
    parser.add_argument("--output_motion_b", required=True, help="Output path for character B motion pkl")
    parser.add_argument("--output_char_a", default=None, help="Optional output path for extracted character A MJCF xml")
    parser.add_argument("--output_char_b", default=None, help="Optional output path for extracted character B MJCF xml")
    parser.add_argument("--scene_xml", default=None, help="Optional override for the dual scene xml")
    parser.add_argument("--loop", default="clamp", choices=["wrap", "clamp"], help="Loop mode")
    parser.add_argument("--fps_key", default="fps", help="Preferred key for fps lookup")
    parser.add_argument("--qpos_key_a", default=None, help="Override key for A qpos array")
    parser.add_argument("--qpos_key_b", default=None, help="Override key for B qpos array")
    parser.add_argument("--root_pos_key_a", default=None, help="Override key for A root positions")
    parser.add_argument("--root_pos_key_b", default=None, help="Override key for B root positions")
    parser.add_argument("--root_rot_key_a", default=None, help="Override key for A root quaternions")
    parser.add_argument("--root_rot_key_b", default=None, help="Override key for B root quaternions")
    parser.add_argument("--dof_pos_key_a", default=None, help="Override key for A dof positions")
    parser.add_argument("--dof_pos_key_b", default=None, help="Override key for B dof positions")
    parser.add_argument("--char_file_a", default=None, help="Optional target rig for validating motion A width")
    parser.add_argument("--char_file_b", default=None, help="Optional target rig for validating motion B width")
    parser.add_argument("--input_fps", type=int, default=-1, help="Fallback fps when bundle metadata is missing")
    parser.add_argument(
        "--quat_format",
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion layout used by root rotations in the input bundle",
    )
    parser.add_argument(
        "--root_basis_correction",
        default="interx",
        choices=["interx", "none"],
        help="Apply InterX-style root basis correction before writing MimicKit frames",
    )
    args = parser.parse_args()

    convert_dual_bundle(
        input_file=args.input_file,
        output_motion_a=args.output_motion_a,
        output_motion_b=args.output_motion_b,
        output_char_a=args.output_char_a,
        output_char_b=args.output_char_b,
        scene_xml=args.scene_xml,
        loop_mode=args.loop,
        fps_key=args.fps_key,
        qpos_key_a=args.qpos_key_a,
        qpos_key_b=args.qpos_key_b,
        root_pos_key_a=args.root_pos_key_a,
        root_pos_key_b=args.root_pos_key_b,
        root_rot_key_a=args.root_rot_key_a,
        root_rot_key_b=args.root_rot_key_b,
        dof_pos_key_a=args.dof_pos_key_a,
        dof_pos_key_b=args.dof_pos_key_b,
        char_file_a=args.char_file_a,
        char_file_b=args.char_file_b,
        input_fps=args.input_fps,
        quat_format=args.quat_format,
        root_basis_correction=args.root_basis_correction,
    )


if __name__ == "__main__":
    main()
