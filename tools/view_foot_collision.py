#!/usr/bin/env python3
"""足裏の接触形状（foot_collision カプセル）を MuJoCo で描いて確認する。

衝突形状は group 3、見た目のメッシュは group 2 に分かれている。
既定のビューアは group 3 を隠すので、そのままでは足裏の当たり判定が見えない。
このツールは可視グループを明示して、静止画（4 面図）と対話ビューアの両方を出す。

    python3 tools/view_foot_collision.py                 # outputs/foot_collision.png
    python3 tools/view_foot_collision.py --interactive    # ビューアで回して見る
    python3 tools/view_foot_collision.py --mode overlay   # メッシュを透過してカプセルを重ねる

姿勢は crouch（isaac_policy_config.h の kCrouchJointPositionRad）。
ピッチ面では足裏が水平になる姿勢。ロールは hip_roll = ∓12 deg ぶん傾いたままになる。
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "robotmodel" / "big_rabbit" / "scene.xml"
POLICY_CONFIG = REPO_ROOT / "dep" / "motion_control" / "include" / "isaac_policy_config.h"

# 注目させたい geom は色を変える。左右で分けると fromto の向きが読み取れる。
FOOT_COLLISION_RGBA = {
    "left_foot_collision": (1.00, 0.55, 0.10, 1.0),
    "right_foot_collision": (0.20, 0.75, 1.00, 1.0),
}

# 接地ちょうど（dist = 0）だと接触点が片端しか出ない。0.3 mm 沈めて線接触を見えるようにする。
SINK_M = 0.0003

HIDDEN_GROUP = 5


def _config_text() -> str:
    return POLICY_CONFIG.read_text()


def read_crouch_pose() -> np.ndarray:
    """crouch 姿勢を C++ ヘッダから読む。姿勢の正本は生成ヘッダ側なので複製しない。"""
    match = re.search(r"kCrouchJointPositionRad\s*=\s*\{(.*?)\};", _config_text(), re.S)
    if match is None:
        raise SystemExit(f"kCrouchJointPositionRad が読めない: {POLICY_CONFIG}")
    values = re.findall(r"([+-]?\d+\.\d+)f", match.group(1))
    if len(values) != 10:
        raise SystemExit(f"crouch 姿勢の要素数が 10 でない: {len(values)}")
    return np.array([float(v) for v in values], dtype=np.float64)


def read_spawn_height() -> float:
    match = re.search(r"kSpawnHeightM\s*=\s*([+-]?\d+\.\d+)f", _config_text())
    if match is None:
        raise SystemExit(f"kSpawnHeightM が読めない: {POLICY_CONFIG}")
    return float(match.group(1))


def joint_names() -> list[str]:
    match = re.search(r"kJointNames\s*=\s*\{(.*?)\};", _config_text(), re.S)
    if match is None:
        raise SystemExit(f"kJointNames が読めない: {POLICY_CONFIG}")
    return re.findall(r'"([^"]+)"', match.group(1))


def setup_pose(mujoco, model, data) -> None:
    """crouch 姿勢を入れて、足裏が床にちょうど触る高さまで骨盤を落とす。"""
    data.qpos[:] = model.qpos0
    data.qpos[2] = read_spawn_height()
    for name, angle in zip(joint_names(), read_crouch_pose()):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    mujoco.mj_forward(model, data)

    # 足裏カプセルの最下点を床（z=0）に合わせる。接触の見え方を安定させるため。
    lowest = min(
        data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)][2]
        - model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)][0]
        for name in FOOT_COLLISION_RGBA
    )
    data.qpos[2] -= lowest + SINK_M
    mujoco.mj_forward(model, data)


def scale_contact_markers(model) -> None:
    """接触点 / 力の描画寸法。既定は stat.extent 基準で足に対して大きすぎる。"""
    model.vis.scale.contactwidth = 0.015
    model.vis.scale.contactheight = 0.004
    model.vis.scale.forcewidth = 0.004
    model.vis.map.force = 0.003


def paint(mujoco, model, mode: str) -> None:
    """mode に応じて色と透過を仕込む。XML は触らず、読み込んだ model 側だけを変える。"""
    for name, rgba in FOOT_COLLISION_RGBA.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_rgba[geom_id] = rgba
    if mode == "overlay":
        # 見た目メッシュは material 経由で着色されるので、mat_rgba の alpha を落とす。
        model.mat_rgba[:, 3] = 0.22


def hide_other_collision(mujoco, model, hide: bool) -> None:
    """近接図では脚・胴の当たり判定ボックスが視界を塞ぐので、足以外を隠す。"""
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name in FOOT_COLLISION_RGBA or name == "floor":
            continue
        if model.geom_group[geom_id] in (3, HIDDEN_GROUP):
            model.geom_group[geom_id] = HIDDEN_GROUP if hide else 3


def visible_groups(mode: str) -> list[int]:
    if mode == "visual":
        return [0, 1, 2]
    if mode == "collision":
        return [0, 1, 3]
    return [0, 1, 2, 3]  # overlay


def make_scene_option(mujoco, mode: str, label: bool):
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[:] = 0
    for group in visible_groups(mode):
        option.geomgroup[group] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    if label:
        option.label = mujoco.mjtLabel.mjLABEL_GEOM
    return option


def mesh_extent(mujoco, model, data, geom_name: str) -> np.ndarray:
    """メッシュ geom の world AABB。見た目の足と当たり判定カプセルを比べるため。"""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
    world = data.geom_xpos[geom_id] + vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
    return np.array([world.min(axis=0), world.max(axis=0)])


def capsule_report(mujoco, model, data) -> str:
    lines = []
    for name in FOOT_COLLISION_RGBA:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        radius, half_length = model.geom_size[geom_id][:2]
        center = data.geom_xpos[geom_id]
        axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]  # カプセル軸は geom の z 軸
        end_a = center - axis * half_length
        end_b = center + axis * half_length
        lines.append(
            f"{name}\n"
            f"  type          = capsule\n"
            f"  radius        = {radius * 1000:.1f} mm\n"
            f"  axis length   = {half_length * 2 * 1000:.1f} mm (総長 {(half_length * 2 + 2 * radius) * 1000:.1f} mm)\n"
            f"  world end A   = [{end_a[0]:+.4f} {end_a[1]:+.4f} {end_a[2]:+.4f}]\n"
            f"  world end B   = [{end_b[0]:+.4f} {end_b[1]:+.4f} {end_b[2]:+.4f}]\n"
            f"  lowest point  = z {(center[2] - radius) * 1000:+.1f} mm\n"
            f"  friction      = {model.geom_friction[geom_id]}\n"
            f"  contype/aff   = {model.geom_contype[geom_id]}/{model.geom_conaffinity[geom_id]}"
        )
    # 見た目のメッシュとの比較。足裏は矩形の footprint を持つが、当たり判定は中心線 1 本しかない。
    box = mesh_extent(mujoco, model, data, "left_foot_visual_0")
    capsule_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    radius, half_length = model.geom_size[capsule_id][:2]
    center = data.geom_xpos[capsule_id]
    lines.append(
        "左足 見た目メッシュ (FOOT_LINK_B) の world AABB との比較\n"
        f"  mesh  X {box[0][0]:+.4f} .. {box[1][0]:+.4f}  ({(box[1][0] - box[0][0]) * 1000:.1f} mm)\n"
        f"  mesh  Y {box[0][1]:+.4f} .. {box[1][1]:+.4f}  ({(box[1][1] - box[0][1]) * 1000:.1f} mm)\n"
        f"  mesh  Z {box[0][2]:+.4f} .. {box[1][2]:+.4f}  ({(box[1][2] - box[0][2]) * 1000:.1f} mm)\n"
        f"  capsule X {center[0] - half_length - radius:+.4f} .. {center[0] + half_length + radius:+.4f}"
        f"  ({(half_length + radius) * 2 * 1000:.1f} mm)\n"
        f"  capsule Y {center[1] - radius:+.4f} .. {center[1] + radius:+.4f}  ({radius * 2 * 1000:.1f} mm)\n"
        f"  capsule 最下点 z {center[2] - radius:+.4f} / mesh 最下点 z {box[0][2]:+.4f}"
        f"  -> カプセルが {(box[0][2] - (center[2] - radius)) * 1000:+.1f} mm 下に出ている"
    )

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    lines.append(f"crouch で接地させたときの骨盤高 = {data.xpos[pelvis][2] * 1000:.1f} mm")

    contacts = [
        f"  contact[{i}] pos=[{data.contact[i].pos[0]:+.4f} {data.contact[i].pos[1]:+.4f}"
        f" {data.contact[i].pos[2]:+.4f}] dist={data.contact[i].dist * 1000:+.2f} mm"
        f" geoms={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1)}"
        f"/{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2)}"
        for i in range(data.ncon)
    ]
    lines.append(f"ncon = {data.ncon}" + ("\n" + "\n".join(contacts) if contacts else ""))
    return "\n".join(lines)


def render_panels(mujoco, model, data, option, size: int) -> np.ndarray:
    """左足に寄った 3 面（側面 / 正面 / 真下）＋ 全身俯瞰を 1 枚にまとめる。"""
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    foot_pos = data.geom_xpos[foot_id].copy()

    # (ラベル, lookat, azimuth, elevation, distance, 足以外の当たり判定を隠すか)
    views = [
        ("side (X-Z)", foot_pos, -90.0, -6.0, 0.34, True),
        ("front (Y-Z)", foot_pos, 180.0, -6.0, 0.34, True),
        ("bottom", foot_pos, -90.0, 70.0, 0.34, True),
        ("whole body", np.array([0.0, 0.0, 0.20]), 135.0, -18.0, 1.10, False),
    ]

    tiles = []
    with mujoco.Renderer(model, height=size, width=size) as renderer:
        for name, lookat, azimuth, elevation, distance, hide in views:
            hide_other_collision(mujoco, model, hide)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = distance
            renderer.update_scene(data, camera, scene_option=option)
            tile = renderer.render().astype(np.uint8)
            tile[:2, :] = tile[-2:, :] = tile[:, :2] = tile[:, -2:] = 255  # 枠線
            tiles.append(tile)
            print(f"  panel: {name}")
    hide_other_collision(mujoco, model, False)
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def launch_viewer(mujoco, model, data, option) -> None:
    import mujoco.viewer

    print("ビューア起動。マウス左ドラッグで回転 / 右ドラッグで平行移動 / ホイールでズーム。")
    print("キー 2 で見た目メッシュ、3 で衝突形状の表示を切り替えられる。")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.opt.geomgroup[:] = option.geomgroup
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = data.geom_xpos[foot_id]
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 120.0, -15.0, 0.45
        # 物理は進めない。形状を見るだけなので姿勢を固定したまま描画だけ回す。
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


def save_image(image: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(image).save(out)
    except ImportError:
        import imageio.v3 as iio

        iio.imwrite(out, image)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--mode",
        choices=("collision", "visual", "overlay"),
        default="collision",
        help="collision: 当たり判定だけ / visual: 見た目メッシュだけ / overlay: メッシュ透過に重ねる",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "foot_collision.png")
    parser.add_argument("--size", type=int, default=560, help="1 パネルの辺 [px]")
    parser.add_argument("--label", action="store_true", help="geom 名を画面に出す")
    parser.add_argument("--interactive", action="store_true", help="静止画の代わりにビューアを開く")
    args = parser.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    paint(mujoco, model, args.mode)
    scale_contact_markers(model)
    setup_pose(mujoco, model, data)
    print(capsule_report(mujoco, model, data))

    option = make_scene_option(mujoco, args.mode, args.label)
    if args.interactive:
        launch_viewer(mujoco, model, data, option)
        return 0

    image = render_panels(mujoco, model, data, option, args.size)
    save_image(image, args.out)
    print(f"wrote {args.out}  ({image.shape[1]}x{image.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
