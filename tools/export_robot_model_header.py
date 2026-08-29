#!/usr/bin/env python3
"""MJCF から RobotCalc 形式のロボットモデルヘッダを作る。

出力: dep/motion_control/include/big_rabbit_robot_model.h

接地判定と骨盤高の推定を実機でやるには、bridge の中に運動学が要る。
その運動学モデルを **MJCF（robotmodel/big_rabbit/big_rabbit.xml）から機械的に落とす**。
MJCF は Isaac の USD と同じ生成元を持つ正本なので、こうしておけば
「sim と実機で別のモデルを持つ」事故が起きない。

構造体は RobotCalc（/home/pomiou/work/RobotCalc/src/RCL_RobotModel.h）の link / robot を
そのまま使う。RNEA などの既存コードを載せ替えずに使うため。ただし 1 点だけ拡張がある:

    RobotCalc の前提はフレーム間が「平行移動のみ」で、RCL_Dynamics.cc:61 が
        r->l[i].R = rot33(r->l[i].a_joint, q(i));
    と R を関節回転だけで上書きする。Big Rabbit はリンク間に固定回転があるので
    表現できない。そこで link に R_fixed（親->自リンクの固定回転）を足し、
        r->l[i].R = r->l[i].R_fixed * rot33(r->l[i].a_joint, q(i));
    で使う。既存コードを流用するときはこの 1 行だけ直す。

脚は左右で独立した 5 関節の直列鎖として出す（base = 骨盤）。浮遊ベースの扱いは
呼ぶ側の仕事。関節角は **MJCF / Isaac / policy と同じ定義**（bridge の
g_control_info.joint_position がそのまま入る）。モータ側の符号・減速比は
big_rabbit_model_param.h の担当で、ここには出てこない。

    python3 tools/export_robot_model_header.py            # 生成して自己検証
    python3 tools/export_robot_model_header.py --check    # 生成せず検証だけ
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = REPO_ROOT / "robotmodel" / "big_rabbit" / "big_rabbit.xml"
DEFAULT_OUT = REPO_ROOT / "dep" / "motion_control" / "include" / "big_rabbit_robot_model.h"
POLICY_CONFIG = REPO_ROOT / "dep" / "motion_control" / "include" / "isaac_policy_config.h"
TEMPLATE = REPO_ROOT / "tools" / "templates" / "big_rabbit_robot_model.h.in"
STAMP_PREFIX = "// generator-stamp sha256: "
# 検証ビルドに使う Eigen。repo 同梱（dep/Eigen）を優先し、無ければ RobotCalc のものを借りる。
EIGEN_CANDIDATES = [REPO_ROOT / "dep", Path("/home/pomiou/work/RobotCalc/src")]


def find_eigen() -> Path | None:
    for candidate in EIGEN_CANDIDATES:
        if (candidate / "Eigen" / "Dense").exists():
            return candidate
    return None

LEGS = {
    "left": ["pelvis", "left_hip_yaw", "left_hip_roll", "left_thigh", "left_shin", "left_foot"],
    "right": ["pelvis", "right_hip_yaw", "right_hip_roll", "right_thigh", "right_shin",
              "right_foot"],
}
JOINTS = {
    "left": ["left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
             "left_knee_joint", "left_ankle_joint"],
    "right": ["right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
              "right_knee_joint", "right_ankle_joint"],
}
AXIS_ENUM = {(1, 0, 0): "x", (0, 1, 0): "y", (0, 0, 1): "z",
             (-1, 0, 0): "minus_x", (0, -1, 0): "minus_y", (0, 0, -1): "minus_z"}


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """MuJoCo の (w,x,y,z) -> 3x3。返る R は「自リンク -> 親リンク」（列が自リンクの軸）。"""
    import mujoco
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=np.float64))
    return matrix.reshape(3, 3)


def axis_name(axis: np.ndarray) -> str:
    key = tuple(int(round(value)) for value in axis)
    if key not in AXIS_ENUM or not np.allclose(axis, key, atol=1e-9):
        raise SystemExit(f"関節軸が座標軸に乗っていない: {axis}。a_joint enum を拡張する必要がある")
    return AXIS_ENUM[key]


def read_policy_config() -> tuple[np.ndarray, float]:
    """crouch 姿勢と spawn 高さを生成ヘッダから読む（検証にだけ使う）。"""
    text = POLICY_CONFIG.read_text(encoding="utf-8")
    spawn = float(re.search(r"kSpawnHeightM\s*=\s*([-0-9.eEf+]+?)f?;", text).group(1))
    block = re.search(r"kCrouchJointPositionRad\s*=\s*\{(.*?)\};", text, re.S).group(1)
    crouch = [float(value) for value in re.findall(r"([-+]?\d+\.\d+)f", block)]
    if len(crouch) != 10:
        raise SystemExit("crouch の読み取りに失敗した")
    return np.array(crouch), spawn


def extract(mjcf: Path) -> dict:
    """MJCF から左右脚のリンク情報と足裏カプセルを取り出す。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    name2id = lambda kind, name: mujoco.mj_name2id(model, kind, name)
    body_of = lambda name: name2id(mujoco.mjtObj.mjOBJ_BODY, name)

    data = {"legs": {}, "total_mass": float(model.body_mass.sum())}
    for side, chain in LEGS.items():
        links = []
        for index, body_name in enumerate(chain):
            body_id = body_of(body_name)
            child_id = body_of(chain[index + 1]) if index + 1 < len(chain) else None
            # 慣性テンソルは MuJoCo が主軸表現で持つ。リンク座標に戻す: R diag(I) R^T
            principal_rotation = quat_to_matrix(model.body_iquat[body_id])
            inertia = principal_rotation @ np.diag(model.body_inertia[body_id]) @ \
                principal_rotation.T
            entry = {
                "body": body_name,
                "mass": float(model.body_mass[body_id]),
                "s_com": model.body_ipos[body_id].copy(),
                "I": inertia,
                # 親 -> 自リンクの固定回転（関節回転を掛ける前）。base は単位行列。
                "R_fixed": quat_to_matrix(model.body_quat[body_id]) if index > 0 else np.eye(3),
                # 自リンク座標で見た子リンク原点。先端リンクは足裏中心（外力作用点）を入れる。
                "p_child": model.body_pos[child_id].copy() if child_id is not None else None,
            }
            if index > 0:
                joint_name = JOINTS[side][index - 1]
                joint_id = name2id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise SystemExit(f"{joint_name} が回転関節でない")
                if not np.allclose(model.jnt_pos[joint_id], 0.0, atol=1e-12):
                    raise SystemExit(f"{joint_name} の関節原点がリンク原点にない。"
                                     "RobotCalc の link 構造体では表現できない")
                entry["joint"] = joint_name
                entry["e_joint"] = model.jnt_axis[joint_id].copy()
                entry["a_joint"] = axis_name(model.jnt_axis[joint_id])
                entry["range"] = model.jnt_range[joint_id].copy()
            links.append(entry)

        # 足裏の衝突カプセル（足リンク座標）。接地判定と骨盤高はこれを地面に当てる。
        geom_id = name2id(mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise SystemExit(f"{side}_foot_collision がカプセルでない")
        radius = float(model.geom_size[geom_id][0])
        half_length = float(model.geom_size[geom_id][1])
        capsule_rotation = quat_to_matrix(model.geom_quat[geom_id])
        center = model.geom_pos[geom_id].copy()
        axis = capsule_rotation[:, 2]  # カプセルの軸は局所 z
        links[-1]["sole"] = {
            "p_a": center - axis * half_length,
            "p_b": center + axis * half_length,
            "center": center,
            "radius": radius,
        }
        data["legs"][side] = links
    return data


# --------------------------------------------------------------- 検証（numpy 側 FK）


def leg_forward_kinematics(links: list[dict], q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """出力する数値だけを使って 骨盤 -> 足リンク の同次変換を組む。

    MuJoCo と同じ合成:  X_parent_child = T(p_child_parent) * R_fixed * Rot(axis, q)
    R は「自リンク -> 親リンク」なので、そのまま右から掛けていける。
    """
    rotation = np.eye(3)
    position = np.zeros(3)
    for index in range(1, len(links)):
        parent = links[index - 1]
        link = links[index]
        position = position + rotation @ parent["p_child"]
        angle = float(q[index - 1])
        axis = np.asarray(link["e_joint"], dtype=np.float64)
        cross = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
        joint_rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
        rotation = rotation @ link["R_fixed"] @ joint_rotation
    return rotation, position


def verify(model_data: dict, mjcf: Path, samples: int) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    crouch, spawn_height = read_policy_config()
    joint_qpos_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                      for name in JOINTS["left"] + JOINTS["right"]]
    root_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]

    rng = np.random.default_rng(20260829)
    max_position_error = 0.0
    max_rotation_error = 0.0
    for trial in range(samples):
        q_all = crouch if trial == 0 else rng.uniform(-1.5, 1.5, size=10)
        data.qpos[:] = model.qpos0
        data.qpos[root_adr:root_adr + 3] = 0.0
        data.qpos[root_adr + 3:root_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        for index, adr in enumerate(joint_qpos_adr):
            data.qpos[adr] = q_all[index]
        mujoco.mj_forward(model, data)
        for leg_index, side in enumerate(("left", "right")):
            links = model_data["legs"][side]
            rotation, position = leg_forward_kinematics(
                links, q_all[leg_index * 5:(leg_index + 1) * 5])
            foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
            max_position_error = max(max_position_error,
                                     float(np.abs(position - data.xpos[foot_id]).max()))
            max_rotation_error = max(max_rotation_error,
                                     float(np.abs(rotation - data.xmat[foot_id].reshape(3, 3)).max()))
    print(f"[検証] 骨盤->足リンク FK と MuJoCo の差（{samples} 姿勢）: "
          f"位置 {max_position_error:.3e} m / 回転 {max_rotation_error:.3e}")
    if max_position_error > 1.0e-6 or max_rotation_error > 1.0e-6:
        raise SystemExit("FK が MuJoCo と合わない。移植が壊れている")

    # 足裏カプセルの最下点。crouch で骨盤を spawn 高さに置けば、ほぼ地面に接するはず。
    data.qpos[:] = model.qpos0
    data.qpos[root_adr:root_adr + 3] = [0.0, 0.0, spawn_height]
    data.qpos[root_adr + 3:root_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    for index, adr in enumerate(joint_qpos_adr):
        data.qpos[adr] = crouch[index]
    mujoco.mj_forward(model, data)
    expected = {"crouch": crouch, "foot": {}, "sole": {}, "axisa": {}, "axisb": {}, "jac": {}}
    pelvis_offset = np.array([0.0, 0.0, spawn_height])
    for leg_index, side in enumerate(("left", "right")):
        links = model_data["legs"][side]
        rotation, position = leg_forward_kinematics(links, crouch[leg_index * 5:(leg_index + 1) * 5])
        sole = links[-1]["sole"]
        lowest = min(
            (position + pelvis_offset + rotation @ sole[key])[2] - sole["radius"]
            for key in ("p_a", "p_b"))
        print(f"[検証] {side:5s} 足裏最下点の world z（crouch, 骨盤 {spawn_height:.3f} m）: "
              f"{lowest:+.5f} m")

        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
        foot_rotation = data.xmat[foot_id].reshape(3, 3)
        # 骨盤基準（骨盤は原点・無回転に置いてある）へ直す
        expected["foot"][side] = data.xpos[foot_id] - pelvis_offset
        contact_point = sole["center"].copy()
        contact_point[2] -= sole["radius"]
        expected["sole"][side] = expected["foot"][side] + foot_rotation @ contact_point
        expected["axisa"][side] = expected["foot"][side] + foot_rotation @ sole["p_a"]
        expected["axisb"][side] = expected["foot"][side] + foot_rotation @ sole["p_b"]

        # 足裏中心での幾何ヤコビ。MuJoCo に出させて脚 5 自由度の列だけ取る。
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, expected["sole"][side] + pelvis_offset, foot_id)
        dof = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
               for name in JOINTS[side]]
        expected["jac"][side] = np.vstack([jacp[:, dof], jacr[:, dof]])

    expected["com"] = data.subtree_com[0] - pelvis_offset
    expected["mass"] = float(model.body_mass.sum())
    return expected


# ------------------------------------------------------------------- 出力


def float_literal(value: float) -> str:
    """C++ の float リテラル。'0f' は不正なので必ず小数点か指数を残す。"""
    text = f"{float(value):+.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def vector_literal(vector: np.ndarray) -> str:
    return ", ".join(float_literal(value) for value in vector)


def matrix_literal(matrix: np.ndarray) -> str:
    return ", ".join(float_literal(value) for row in matrix for value in row)


def provenance_lines(model_data: dict, mjcf: Path) -> list[str]:
    lines = [
        "tools/export_robot_model_header.py が生成。**このファイルを直接編集しない**。",
        f"  数値の出どころ: {mjcf.relative_to(REPO_ROOT)}"
        "（Isaac の USD と同じ MJCF。sim と実機で同じモデルを使うため）",
        f"  ロジックの原本: {TEMPLATE.relative_to(REPO_ROOT)}"
        "（ここを直して再生成する）",
        "",
        "用途: 実機での接地判定と骨盤高の推定に使う運動学モデル。",
        "構造体は RobotCalc（RCL_RobotModel.h）の link / robot をそのまま踏襲する。",
        "ただしリンク間に固定回転があるので link に R_fixed を足してある。",
        "RobotCalc の RNEA / 動力学を流用するときは、回転行列の更新を",
        "    r->l[i].R = r->l[i].R_fixed * rot33(r->l[i].a_joint, q(i));",
        "に変える（元は rot33 だけで上書きしている）。BigRabbitRobotUpdate はこの形で回す。",
        "",
        "関節角の定義は MJCF / Isaac / policy と同一。bridge の joint_position がそのまま入る。",
        "モータ次元の符号・減速比は big_rabbit_model_param.h の担当でここには出てこない。",
        "",
        "脚は左右独立の 5 関節直列鎖（base = 骨盤）。浮遊ベースの扱いは呼ぶ側。",
        "",
        "リンク構成:",
    ]
    for side, links in model_data["legs"].items():
        lines.append(f"  {side:5s}: " + " -> ".join(link["body"] for link in links))
    lines.append(f"全質量（骨盤 + 両脚）: {model_data['total_mass']:.6f} kg")
    return lines


def data_lines(model_data: dict) -> list[str]:
    lines: list[str] = []
    add = lines.append
    add("namespace big_rabbit_model_data")
    add("{")
    add("    struct LinkData")
    add("    {")
    add("        float p_child[3];  // 自リンク座標での子リンク原点（先端は足裏中心）")
    add("        float R_fixed[9];  // 親 <- 自リンク の固定回転（行優先）")
    add("        float e_joint[3];  // 自リンク座標での関節軸")
    add("        int a_joint;       // enum axis")
    add("        float s_com[3];    // 自リンク座標での重心")
    add("        float m;           // 質量 [kg]")
    add("        float I[9];        // 自リンク座標・重心まわりの慣性テンソル（行優先）")
    add("    };")
    add("")
    add(f"    inline constexpr float kTotalMass = {float_literal(model_data['total_mass'])};")
    add("")

    for side, links in model_data["legs"].items():
        name = side.capitalize()
        base = links[0]
        add(f"    // ---- {side} leg: base = {base['body']} ----")
        add(f"    inline constexpr LinkData k{name}Base = {{")
        add(f"        {{{vector_literal(base['p_child'])}}},  // p_child: "
            f"{links[1]['body']} の原点（骨盤座標）")
        add("        {+1.0f, +0.0f, +0.0f, +0.0f, +1.0f, +0.0f, +0.0f, +0.0f, +1.0f},"
            "  // R_fixed: base は単位行列")
        add("        {+0.0f, +0.0f, +0.0f}, none,  // base に関節は無い")
        add(f"        {{{vector_literal(base['s_com'])}}},")
        add(f"        {float_literal(base['mass'])},")
        add(f"        {{{matrix_literal(base['I'])}}},")
        add("    };")
        add("")
        add(f"    inline constexpr LinkData k{name}Link[BIG_RABBIT_LEG_JOINT_NUM] = {{")
        for index in range(1, len(links)):
            link = links[index]
            tip = index == len(links) - 1
            child_comment = ("足裏中心（外力作用点）" if tip
                             else f"{links[index + 1]['body']} の原点")
            p_child = link["p_child"]
            if tip:
                sole = link["sole"]
                p_child = sole["center"].copy()
                p_child[2] -= sole["radius"]  # 軸から半径だけ下ろした接地面上の点
            add(f"        {{ // l[{index - 1}] {link['body']}  ({link['joint']}, "
                f"可動 {np.degrees(link['range'][0]):+.0f}..{np.degrees(link['range'][1]):+.0f} deg)")
            add(f"            {{{vector_literal(p_child)}}},  // p_child: {child_comment}")
            add(f"            {{{matrix_literal(link['R_fixed'])}}},")
            add(f"            {{{vector_literal(link['e_joint'])}}}, {link['a_joint']},")
            add(f"            {{{vector_literal(link['s_com'])}}},")
            add(f"            {float_literal(link['mass'])},")
            add(f"            {{{matrix_literal(link['I'])}}},")
            add("        },")
        add("    };")
        add("")
        sole = links[-1]["sole"]
        add(f"    // {side} 足裏カプセル（{links[-1]['body']} 座標）")
        add(f"    inline constexpr float k{name}SoleA[3] = {{{vector_literal(sole['p_a'])}}};")
        add(f"    inline constexpr float k{name}SoleB[3] = {{{vector_literal(sole['p_b'])}}};")
        add(f"    inline constexpr float k{name}SoleRadius = {float_literal(sole['radius'])};")
        add("")
    add("} // namespace big_rabbit_model_data")
    return lines


def stamp_of(text: str) -> str:
    """スタンプ行を除いた本文のハッシュ。手編集の検出に使う。"""
    body = "\n".join(line for line in text.split("\n") if not line.startswith(STAMP_PREFIX))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def emit(model_data: dict, mjcf: Path, out: Path, force: bool) -> None:
    if out.exists():
        current = out.read_text(encoding="utf-8")
        recorded = next((line[len(STAMP_PREFIX):].strip() for line in current.split("\n")
                         if line.startswith(STAMP_PREFIX)), None)
        if recorded is None:
            print(f"[警告] {out.name} にスタンプが無い（旧版か手書き）。上書きする")
        elif recorded != stamp_of(current) and not force:
            raise SystemExit(
                f"{out.name} が生成後に手で編集されている。上書きすると失われる。\n"
                f"  直したいのは中身なら {TEMPLATE.relative_to(REPO_ROOT)} を編集して再生成する。\n"
                f"  それでも上書きしてよければ --force を付ける。")

    template = TEMPLATE.read_text(encoding="utf-8")
    if "// @@PROVENANCE@@" not in template or "// @@DATA@@" not in template:
        raise SystemExit("テンプレートに @@PROVENANCE@@ / @@DATA@@ が無い")
    text = template.replace(
        "// @@PROVENANCE@@",
        "\n".join(f"// {line}".rstrip() for line in provenance_lines(model_data, mjcf)))
    text = text.replace("// @@DATA@@", "\n".join(data_lines(model_data)))
    text = text.replace("#pragma once\n", f"#pragma once\n\n{STAMP_PREFIX}@@STAMP@@\n", 1)
    text = text.replace("@@STAMP@@", stamp_of(text))
    out.write_text(text, encoding="utf-8")
    print(f"書いた: {out}  ({len(text.splitlines())} 行)")


def build_and_check(out: Path, expected: dict) -> None:
    """生成した **C++ を実際にビルドして走らせ**、MuJoCo と突き合わせる。

    numpy 側の検証だけだと「表は正しいが BigRabbitRobotUpdate の合成規約が違う」を
    見逃す。ヤコビと重心まで C++ 側の出力で確かめる。
    """
    eigen_dir = find_eigen()
    if eigen_dir is None:
        print("[C++ 検証] Eigen が見つからないので飛ばす")
        return
    crouch = expected["crouch"]
    source = out.parent / "_model_fk_check.cc"
    binary = out.parent / "_model_fk_check"
    source.write_text(
        f'#include "{out.name}"\n'
        "#include <cstdio>\n"
        "\n"
        "int main()\n"
        "{\n"
        "    const float q[10] = {" + ", ".join(f"{value:+.9f}f" for value in crouch) + "};\n"
        "    BigRabbitRobotUpdate(q);\n"
        "    const BigRabbitState &s = BigRabbitRobotState();\n"
        "    for (int side = 0; side < BIG_RABBIT_LEG_NUM; side++) {\n"
        "        const BigRabbitLegState &leg = s.state[side];\n"
        "        std::printf(\"foot %d %.9f %.9f %.9f\\n\", side,\n"
        "                    leg.p_foot_base.x(), leg.p_foot_base.y(), leg.p_foot_base.z());\n"
        "        std::printf(\"sole %d %.9f %.9f %.9f\\n\", side,\n"
        "                    leg.p_sole_base.x(), leg.p_sole_base.y(), leg.p_sole_base.z());\n"
        "        std::printf(\"axisa %d %.9f %.9f %.9f\\n\", side,\n"
        "                    leg.p_axis_a_base.x(), leg.p_axis_a_base.y(), leg.p_axis_a_base.z());\n"
        "        std::printf(\"axisb %d %.9f %.9f %.9f\\n\", side,\n"
        "                    leg.p_axis_b_base.x(), leg.p_axis_b_base.y(), leg.p_axis_b_base.z());\n"
        "        for (int row = 0; row < 6; row++) {\n"
        "            std::printf(\"jac %d %d\", side, row);\n"
        "            for (int col = 0; col < BIG_RABBIT_LEG_JOINT_NUM; col++)\n"
        "                std::printf(\" %.9f\", leg.J_sole(row, col));\n"
        "            std::printf(\"\\n\");\n"
        "        }\n"
        "    }\n"
        "    std::printf(\"com %.9f %.9f %.9f\\n\", s.com_base.x(), s.com_base.y(), s.com_base.z());\n"
        "    std::printf(\"mass %.9f\\n\", s.mass_total);\n"
        "    return 0;\n"
        "}\n", encoding="utf-8")
    try:
        build = subprocess.run(
            ["g++", "-std=c++17", "-O2", f"-I{eigen_dir}", f"-I{out.parent}",
             str(source), "-o", str(binary)], capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stdout + build.stderr)
            raise SystemExit("生成ヘッダのビルドに失敗した")
        run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)

        worst = {"foot": 0.0, "sole": 0.0, "axisa": 0.0, "axisb": 0.0, "jac": 0.0}
        for row in run.stdout.splitlines():
            values = row.split()
            if values[0] in ("foot", "sole", "axisa", "axisb"):
                side = ("left", "right")[int(values[1])]
                got = np.array([float(value) for value in values[2:5]])
                worst[values[0]] = max(worst[values[0]],
                                       float(np.abs(got - expected[values[0]][side]).max()))
            elif values[0] == "jac":
                side = ("left", "right")[int(values[1])]
                jac_row = int(values[2])
                got = np.array([float(value) for value in values[3:]])
                worst["jac"] = max(worst["jac"],
                                   float(np.abs(got - expected["jac"][side][jac_row]).max()))
            elif values[0] == "com":
                got = np.array([float(value) for value in values[1:4]])
                com_error = float(np.abs(got - expected["com"]).max())
            elif values[0] == "mass":
                mass_error = abs(float(values[1]) - expected["mass"])

        print(f"[C++ 検証] BigRabbitRobotUpdate と MuJoCo の差（crouch 姿勢、Eigen: {eigen_dir}）")
        print(f"           足リンク原点 {worst['foot']:.2e} m / 足裏中心 {worst['sole']:.2e} m / "
              f"カプセル端点 {max(worst['axisa'], worst['axisb']):.2e} m")
        print(f"           足裏ヤコビ {worst['jac']:.2e} / 全身重心 {com_error:.2e} m / "
              f"全質量 {mass_error:.2e} kg")
        if max(worst.values()) > 1.0e-5 or com_error > 1.0e-5 or mass_error > 1.0e-5:
            raise SystemExit("C++ 側の出力が MuJoCo と合わない")
    finally:
        source.unlink(missing_ok=True)
        binary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=200, help="FK 検証に使う姿勢の数")
    parser.add_argument("--check", action="store_true", help="生成せず検証だけ")
    parser.add_argument("--force", action="store_true", help="手編集された生成物でも上書きする")
    args = parser.parse_args()

    model_data = extract(args.mjcf)
    expected = verify(model_data, args.mjcf, args.samples)
    if not args.check:
        emit(model_data, args.mjcf, args.out, args.force)
        build_and_check(args.out, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
