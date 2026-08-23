#!/usr/bin/env python3
"""big_rabbit_isaac の学習済み policy を C++ ヘッダへ書き出す。

出力は 3 つ。

    isaac_policy_weights.h    actor の重みと observation normalizer
    isaac_policy_config.h     crouch / action scale / 関節名 / 閾値 / gait 周波数
    isaac_walk_reference.h    歩行参照テーブル（64 サンプル x hip_pitch/knee/ankle）

実機 MCU に同じコードを載せる前提なので、ランタイム依存を持たない
constexpr 配列として出す（ONNX や TorchScript は使わない）。

使い方:

    python3 tools/export_isaac_policy_headers.py \
      --checkpoint logs/rsl_rl/big_rabbit_balance/..._v21_stride014/model_499.pt \
      --params configs/big_rabbit_balance/big_rabbit_walk_v21_stride014_4096_500.params.yaml

パスは --isaac-root からの相対で解決する。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_ROOT = Path("/home/pomiou/work/big_rabbit_isaac")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dep" / "motion_control" / "include"

# 学習側の observation 構成。Isaac Lab の Active Observation Terms 出力が正本。
# ここは自動で取れないので、次元だけ検証して食い違いを弾く。
OBS_LAYOUT = [
    ("base_height", 1),
    ("projected_gravity", 3),
    ("base_ang_vel", 3),
    ("feet_contact", 2),
    ("motion_command", 3),
    ("gait_phase", 2),
    ("joint_pos_rel_to_crouch", 10),
    ("joint_vel", 10),
    ("last_action_2", 10),
    ("actions", 10),
]


def _fmt(value: float) -> str:
    return f"{float(value):.9e}f"


def _array(name: str, values, columns: int = 6) -> str:
    flat = [float(v) for v in np.asarray(values).reshape(-1).tolist()]
    lines = [f"inline constexpr float {name}[{len(flat)}] = {{"]
    for start in range(0, len(flat), columns):
        lines.append("    " + ", ".join(_fmt(v) for v in flat[start : start + columns]) + ",")
    lines.append("};")
    return "\n".join(lines)


def _load_params(isaac_root: Path, params_rel: Path) -> dict:
    """big_rabbit_lab の loader をそのまま使う（merge 規則を一致させるため）。"""
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))
    import os

    params_path = params_rel if params_rel.is_absolute() else (isaac_root / params_rel)
    # params.py は import 時に BIG_RABBIT_PARAMS_FILE を読んで merge する。
    # import より先に環境変数を立てる必要がある。
    os.environ["BIG_RABBIT_PARAMS_FILE"] = str(params_path.resolve())
    from big_rabbit_lab.config.params import PARAMS  # noqa: E402

    return PARAMS


def _require(state: dict, key: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"checkpoint に {key} が無い")
    return state[key].detach().cpu().to(dtype=torch.float32).contiguous()


def export_weights(checkpoint: Path, output: Path, params_path: Path,
                   gait_frequency_hz: float) -> tuple[int, int, int]:
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = data.get("model_state_dict")
    if state is None:
        raise KeyError("checkpoint に model_state_dict が無い")

    tensors = {
        "kActor0Weight": _require(state, "actor.0.weight"),
        "kActor0Bias": _require(state, "actor.0.bias"),
        "kActor2Weight": _require(state, "actor.2.weight"),
        "kActor2Bias": _require(state, "actor.2.bias"),
        "kActor4Weight": _require(state, "actor.4.weight"),
        "kActor4Bias": _require(state, "actor.4.bias"),
        "kActor6Weight": _require(state, "actor.6.weight"),
        "kActor6Bias": _require(state, "actor.6.bias"),
        "kObsMean": _require(state, "actor_obs_normalizer._mean"),
        "kObsStd": _require(state, "actor_obs_normalizer._std"),
    }

    obs_dim = int(tensors["kActor0Weight"].shape[1])
    action_dim = int(tensors["kActor6Weight"].shape[0])
    hidden_dim = int(tensors["kActor0Weight"].shape[0])

    # 形状が 1 つでも想定と違えば、ここで止める。
    # 黙って通すと obs の並び違いと区別できない不具合になる。
    expected = {
        "kActor0Weight": (hidden_dim, obs_dim),
        "kActor0Bias": (hidden_dim,),
        "kActor2Weight": (hidden_dim, hidden_dim),
        "kActor2Bias": (hidden_dim,),
        "kActor4Weight": (hidden_dim, hidden_dim),
        "kActor4Bias": (hidden_dim,),
        "kActor6Weight": (action_dim, hidden_dim),
        "kActor6Bias": (action_dim,),
        "kObsMean": (1, obs_dim),
        "kObsStd": (1, obs_dim),
    }
    for name, shape in expected.items():
        actual = tuple(tensors[name].shape)
        if actual != shape:
            raise ValueError(f"{name} の形状不一致: 期待 {shape} 実際 {actual}")

    layout_dim = sum(dim for _, dim in OBS_LAYOUT)
    if layout_dim != obs_dim:
        raise ValueError(
            f"OBS_LAYOUT の合計 {layout_dim} と checkpoint の obs 次元 {obs_dim} が違う。"
            " 学習側の観測構成が変わっていないか確認する"
        )

    lines = [
        "#pragma once",
        "",
        "// tools/export_isaac_policy_headers.py が生成。手で編集しないこと。",
        f"// Checkpoint: {checkpoint}",
        f"// Params: {params_path}",
        f"// 構造: obs{obs_dim} -> {hidden_dim} -> {hidden_dim} -> {hidden_dim} -> action{action_dim},"
        " ELU, RSL-RL EmpiricalNormalization",
        "//",
        "// observation の並び（Isaac Lab の Active Observation Terms が正本）:",
    ]
    index = 0
    for name, dim in OBS_LAYOUT:
        span = f"{index}" if dim == 1 else f"{index}-{index + dim - 1}"
        lines.append(f"//   [{span:>7}] {name} ({dim})")
        index += dim
    lines += [
        "",
        "",
        '#include "isaac_policy_config.h"',
        "",
        "namespace isaac_policy {",
        "",
        "// 次元は isaac_policy_config.h 側が正本。食い違えばここで止まる。",
        f"static_assert(kObsDim == {obs_dim}, \"obs 次元が config と checkpoint で違う\");",
        f"static_assert(kActionDim == {action_dim}, \"action 次元が config と checkpoint で違う\");",
        f"static_assert(kHiddenDim == {hidden_dim}, \"hidden 次元が config と checkpoint で違う\");",
        "",
    ]
    for name in (
        "kObsMean",
        "kObsStd",
        "kActor0Weight",
        "kActor0Bias",
        "kActor2Weight",
        "kActor2Bias",
        "kActor4Weight",
        "kActor4Bias",
        "kActor6Weight",
        "kActor6Bias",
    ):
        lines.append(_array(name, tensors[name]))
        lines.append("")
    lines.append("}  // namespace isaac_policy")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return obs_dim, action_dim, hidden_dim


def export_config(params: dict, output: Path, params_path: Path, obs_dim: int, action_dim: int,
                  hidden_dim: int, gait_frequency_hz: float) -> None:
    robot = params["robot"]
    names = list(robot["joint_names"])
    if len(names) != action_dim:
        raise ValueError(f"joint_names {len(names)} と action 次元 {action_dim} が違う")

    crouch = [math.radians(robot["joint_poses_deg"]["crouch"][n]) for n in names]
    scale = [math.radians(robot["joint_action_scale_deg"][n]) for n in names]
    stiffness = [float(robot["joint_stiffness"][n]) for n in names]
    damping = [float(robot["joint_damping"][n]) for n in names]
    effort = [float(robot["actuator"]["effort_limit_sim"][n]) for n in names]
    velocity = [float(robot["actuator"]["velocity_limit_sim"][n]) for n in names]
    contact_threshold = float(params["contact"]["foot_contact_threshold_n"])
    foot_bodies = list(params["contact"]["foot_body_names"])
    spawn_z = float(robot["init_state"]["pos"][2])
    decimation = int(params["environment"]["decimation"])
    physics_dt = float(params["environment"].get("sim_dt", 0.001))

    def block(name: str, values, unit: str) -> list[str]:
        out = [f"    // {unit}", f"    inline constexpr std::array<float, {len(values)}> {name} = {{"]
        for value, joint in zip(values, names):
            out.append(f"        {value:+.10f}f,  // {joint}")
        out.append("    };")
        return out

    lines = [
        "#pragma once",
        "",
        "// tools/export_isaac_policy_headers.py が生成。手で編集しないこと。",
        f"// Params: {params_path}",
        "",
        "#include <array>",
        "#include <cstdlib>",
        "",
        "namespace isaac_policy",
        "{",
        "",
        "    inline constexpr float kPi = 3.14159265358979323846f;",
        "",
        "    // policy の入出力次元。isaac_policy_weights.h が static_assert で突き合わせる。",
        f"    inline constexpr int kObsDim = {obs_dim};",
        f"    inline constexpr int kActionDim = {action_dim};",
        f"    inline constexpr int kHiddenDim = {hidden_dim};",
        "    inline constexpr float kObsNormalizerEps = 1.0e-2f;",
        "",
        "    // 歩容位相の周波数。観測の gait_phase と参照テーブルの位相で共通。",
        f"    inline constexpr float kGaitFrequencyHz = {gait_frequency_hz:.10f}f;",
        "",
        f"    // policy 制御周期。物理 {physics_dt} s x decimation {decimation}。",
        f"    inline constexpr float kStepDt = {decimation * physics_dt:.10f}f;",
        f"    inline constexpr int kDecimation = {decimation};",
        f"    inline constexpr float kPhysicsDt = {physics_dt:.10f}f;",
        "",
        "    // 接地観測の閾値。Isaac の ContactSensor と同じ意味。",
        f"    inline constexpr float kFootContactThresholdN = {contact_threshold:.6f}f;",
        "",
        "    // 起動時の骨盤高。",
        f"    inline constexpr float kSpawnHeightM = {spawn_z:.6f}f;",
        "",
        "    inline constexpr std::array<const char *, %d> kJointNames = {" % len(names),
    ]
    for n in names:
        lines.append(f'        "{n}",')
    lines.append("    };")
    lines.append("")
    lines.append("    inline constexpr std::array<const char *, %d> kFootBodyNames = {" % len(foot_bodies))
    for n in foot_bodies:
        lines.append(f'        "{n}",')
    lines.append("    };")
    lines.append("")
    lines += block("kCrouchJointPositionRad", crouch, "基準姿勢 crouch [rad]。hip+knee+ankle=0 で足裏が水平になる")
    lines.append("")
    lines += block("kJointActionScaleRad", scale, "action 1.0 あたりの関節角 [rad]")
    lines.append("")
    lines += block("kJointStiffness", stiffness, "Isaac の実装 PD 剛性 [N m/rad]（関節側）")
    lines.append("")
    lines += block("kJointDamping", damping, "Isaac の実装 PD 減衰 [N m s/rad]（関節側）")
    lines.append("")
    lines += block("kJointEffortLimitNm", effort, "関節側トルク上限 [N m]")
    lines.append("")
    lines += block("kJointVelocityLimitRadS", velocity, "関節側速度上限 [rad/s]")
    lines += [
        "",
        "    // 指令 (x, y, yaw)。環境変数で上書きできる。",
        "    inline float EnvFloat(const char *name, float default_value)",
        "    {",
        "        const char *value = std::getenv(name);",
        "        if (!value || value[0] == '\\0')",
        "        {",
        "            return default_value;",
        "        }",
        "        char *end = nullptr;",
        "        const float parsed = std::strtof(value, &end);",
        "        return end == value ? default_value : parsed;",
        "    }",
        "",
        "    inline const std::array<float, 3> &MotionCommand()",
        "    {",
        "        static const std::array<float, 3> command = {",
        '            EnvFloat("BIG_RABBIT_MOTION_COMMAND_X", 0.0f),',
        '            EnvFloat("BIG_RABBIT_MOTION_COMMAND_Y", 0.0f),',
        '            EnvFloat("BIG_RABBIT_MOTION_COMMAND_YAW", 0.0f),',
        "        };",
        "        return command;",
        "    }",
        "",
        "}  // namespace isaac_policy",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_reference(params: dict, isaac_root: Path, output: Path) -> None:
    reference = params.get("action_reference", {})
    if not bool(reference.get("enabled", False)):
        raise ValueError("action_reference.enabled が false。位相参照つき policy ではない")

    path = Path(reference["path"])
    if not path.is_absolute():
        path = isaac_root / path
    table = np.load(path)

    names = list(params["robot"]["joint_names"])
    roles = list(reference.get("joint_roles", ["hip_pitch", "knee", "ankle"]))
    # action 順の各要素が参照テーブルのどの列を使うか。使わない要素は -1。
    column_of = {"hip_pitch": 0, "knee": 1, "ankle": 2}
    columns: list[int] = []
    is_right: list[int] = []
    for name in names:
        column = -1
        for role in roles:
            if f"_{role}_" in f"_{name}":
                column = column_of[role]
        columns.append(column)
        is_right.append(1 if name.startswith("right") else 0)

    samples = int(len(table["phases"]))
    series = np.stack(
        [
            np.radians(table["hip_pitch_deg"]),
            np.radians(table["knee_deg"]),
            np.radians(table["ankle_deg"]),
        ],
        axis=1,
    )  # (samples, 3)

    lines = [
        "#pragma once",
        "",
        "// tools/export_isaac_policy_headers.py が生成。手で編集しないこと。",
        f"// 参照テーブル: {path}",
        f"// 歩幅 {float(table['stride_m']):.3f} m / 遊脚高さ {float(table['lift_m']):.3f} m"
        f" / hip_roll {float(table['hip_roll_deg']):.1f} deg / IK 誤差 {float(table['ik_max_error_m']):.2e} m",
        "//",
        "// 位相 0..1 を kReferenceSamples 等分した hip_pitch / knee / ankle の関節角 [rad]。",
        "// 左脚がこの表そのまま、右脚は半周期（kReferenceSamples/2）ずらす。",
        "",
        "#include <array>",
        "",
        "namespace isaac_policy",
        "{",
        "",
        f"    inline constexpr int kReferenceSamples = {samples};",
        f"    inline constexpr int kReferenceHalfShift = {samples // 2};",
        f"    inline constexpr float kReferenceStrideM = {float(table['stride_m']):.6f}f;",
        f"    inline constexpr float kReferenceLiftM = {float(table['lift_m']):.6f}f;",
        "",
        "    // action 順の各要素が使う参照列。-1 は参照を当てず crouch のままにする。",
        f"    inline constexpr std::array<int, {len(names)}> kReferenceColumn = {{",
    ]
    for name, column in zip(names, columns):
        role = "none" if column < 0 else roles[column] if column < len(roles) else str(column)
        lines.append(f"        {column:>2},  // {name} ({role})")
    lines.append("    };")
    lines.append("")
    lines.append(f"    inline constexpr std::array<int, {len(names)}> kReferenceIsRight = {{")
    for name, right in zip(names, is_right):
        lines.append(f"        {right},  // {name}")
    lines.append("    };")
    lines.append("")
    lines.append("    // [sample][hip_pitch, knee, ankle] [rad]")
    lines.append(f"    inline constexpr float kReferenceTableRad[{samples}][3] = {{")
    for row in series:
        lines.append("        {" + ", ".join(_fmt(v) for v in row) + "},")
    lines.append("    };")
    lines.append("")
    lines.append("}  // namespace isaac_policy")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isaac-root", type=Path, default=DEFAULT_ISAAC_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    isaac_root = args.isaac_root.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else isaac_root / args.checkpoint
    params_path = args.params if args.params.is_absolute() else isaac_root / args.params
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint が無い: {checkpoint}")
    if not params_path.is_file():
        raise FileNotFoundError(f"params が無い: {params_path}")

    params = _load_params(isaac_root, params_path)
    gait_frequency_hz = float(params["action_reference"]["frequency_hz"])
    clock_frequency_hz = float(params["gait_clock"]["frequency_hz"])
    if abs(gait_frequency_hz - clock_frequency_hz) > 1.0e-9:
        # 観測の gait_phase と参照の位相は同じ周期でなければならない。
        raise ValueError(
            f"action_reference.frequency_hz {gait_frequency_hz} と "
            f"gait_clock.frequency_hz {clock_frequency_hz} が違う"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = args.output_dir / "isaac_policy_weights.h"
    config = args.output_dir / "isaac_policy_config.h"
    walk_reference = args.output_dir / "isaac_walk_reference.h"

    obs_dim, action_dim, hidden_dim = export_weights(checkpoint, weights, params_path, gait_frequency_hz)
    export_config(params, config, params_path, obs_dim, action_dim, hidden_dim, gait_frequency_hz)
    export_reference(params, isaac_root, walk_reference)

    print(f"[export] obs{obs_dim} / action{action_dim} / gait {gait_frequency_hz} Hz")
    for path in (weights, config, walk_reference):
        print(f"[export] wrote {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
