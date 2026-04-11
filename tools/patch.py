"""Tune SMPLSim MJCF for MimicKit + Newton (gears, ctrlrange, joint PD).

SMPLSim fat exports use gear=1 and stiffness/damping=0. MimicKit's smpl.xml uses
large motor gears and non-zero joint stiffness/damping; Newton copies those into
position-control gains, so fat barely tracked targets without this patch.

Safe to run on:
  data/assets/smpl/human_variants/fat.xml (required for SMPLSim fat)
  data/assets/smpl/smpl.xml — passive pass is skipped if joints already tuned

Usage:
  python tools/patch_fat_actuators.py [path/to/model.xml]
"""
import re
import sys

REPO = __file__.replace("\\", "/").rsplit("/", 2)[0]
path = REPO + "/data/assets/smpl/human_variants/fat.xml"


def gear_for(joint: str) -> int:
    if "Hip" in joint:
        return 300
    if "Knee" in joint:
        return 200
    if "Ankle" in joint:
        return 100
    if "Toe" in joint:
        return 75
    if "Torso" in joint:
        return 300
    if "Spine" in joint:
        return 200
    if "Chest" in joint:
        return 150
    if "Neck" in joint:
        return 75
    if "Head" in joint:
        return 25
    if "Thorax" in joint:
        return 200
    if "Shoulder" in joint:
        return 150
    if "Elbow" in joint:
        return 100
    if "Wrist" in joint:
        return 75
    if "Hand" in joint:
        return 50
    return 25


def passive_triplet(joint_name: str) -> tuple[str, str, str]:
    """(actuatorfrcrange, stiffness, damping) — tiers from data/assets/smpl/smpl.xml."""
    j = joint_name
    if "Hip" in j:
        return ("-300 300", "500", "50")
    if "Knee" in j:
        return ("-200 200", "500", "50")
    if "Ankle" in j:
        return ("-100 100", "400", "40")
    if "Toe" in j:
        return ("-75 75", "200", "20")
    if "Torso" in j:
        return ("-300 300", "1000", "100")
    if "Spine" in j:
        return ("-200 200", "1000", "100")
    if "Chest" in j:
        return ("-150 150", "1000", "100")
    if "Neck" in j:
        return ("-75 75", "500", "50")
    if "Head" in j:
        return ("-25 25", "200", "20")
    if "Thorax" in j:
        return ("-200 200", "500", "50")
    if "Shoulder" in j:
        return ("-150 150", "400", "40")
    if "Elbow" in j:
        return ("-100 100", "300", "30")
    if "Wrist" in j:
        return ("-75 75", "200", "20")
    if "Thumb" in j:
        return ("-50 50", "100", "10")
    return ("-25 25", "200", "20")


def patch_joint_passive(lines: list[str]) -> tuple[list[str], int]:
    """Set actuatorfrcrange + stiffness + damping on hinge joints that are still at zero."""
    out = []
    n = 0
    for line in lines:
        if (
            'type="hinge"' in line
            and 'stiffness="0"' in line
            and 'damping="0"' in line
            and "<joint" in line
        ):
            m = re.search(r'name="([^"]+)"', line)
            if m:
                frc, k, d = passive_triplet(m.group(1))
                line = re.sub(r'\suser="[^"]*"', "", line)
                line = line.replace('armature="0.01"', 'armature="0.02"')
                line = re.sub(
                    r'\s*damping="0"\s*stiffness="0"',
                    f' actuatorfrcrange="{frc}" stiffness="{k}" damping="{d}"',
                    line,
                )
                n += 1
        out.append(line)
    return out, n


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else path
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    pat = re.compile(r'<motor name="([^"]+)" joint="([^"]+)" gear="[0-9]+"/>')

    def repl(m):
        g = gear_for(m.group(2))
        return f'<motor name="{m.group(1)}" joint="{m.group(2)}" gear="{g}"/>'

    text2, n = pat.subn(repl, text)
    print("motors updated:", n)

    lines = text2.splitlines()
    lines, npass = patch_joint_passive(lines)
    text2 = "\n".join(lines)
    if not text2.endswith("\n"):
        text2 += "\n"
    print("hinge passive dynamics updated:", npass)

    if 'ctrlrange="-1 1"' not in text2:
        needle = "  <default>\n    <joint damping="
        insert = '  <default>\n    <motor ctrlrange="-1 1" ctrllimited="true"/>\n    <joint damping='
        if needle in text2:
            text2 = text2.replace(needle, insert, 1)
            print("inserted motor default")
        else:
            print("WARNING: default block pattern not found")

    with open(p, "w", encoding="utf-8") as f:
        f.write(text2)
    print("wrote", p)


if __name__ == "__main__":
    main()
