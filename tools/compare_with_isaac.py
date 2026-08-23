#!/usr/bin/env python3
"""C++ bridge と Isaac 側の obs54 / action10 / 参照姿勢 / 関節目標を突き合わせる。

同じ入力ケースを両方に流し、列ごとに最大差を出す。
これを飛ばすと、歩かない原因が観測の並びなのか制御器なのか切り分けられない。

    python3 tools/compare_with_isaac.py \
      --checkpoint <isaac の model_499.pt> \
      --params configs/big_rabbit_balance/big_rabbit_walk_v21_stride014_4096_500.params.yaml

合格条件（設計書のクロス検証 段 1-3）:
    obs        1e-5 以内
    action     1e-4 以内
    joint_ref  1e-6 以内（参照テーブル補間の一致）
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_ROOT = Path("/home/pomiou/work/big_rabbit_isaac")
JOINT_NUM = 10

# しきい値は設計書のクロス検証 段 1-3 に合わせる。
TOLERANCE = {"obs": 1.0e-5, "action": 1.0e-4, "reference": 1.0e-6, "joint_ref": 1.0e-6, "phase": 1.0e-6}


def build_cases(count: int, seed: int) -> list[list[float]]:
    """位相を 1 周期ぶん均等に走らせつつ、状態と指令を乱数で振ったケースを作る。

    位相を網羅するのが目的。参照テーブルの補間ミスは特定位相でしか出ないことがある。
    """
    rng = np.random.default_rng(seed)
    commands = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.5, 0.0, 0.5],
    ]
    cases: list[list[float]] = []
    for index in range(count):
        # step_count は 0..(1 周期 x 3) を均等に。位相の折り返しも通す。
        step_count = index
        base_height = 0.40 + 0.03 * float(rng.standard_normal())
        # 概ね直立の姿勢 quaternion。小さめの roll/pitch/yaw を乱数で振る。
        # 生値を入れることで quaternion -> 重力方向の変換も検証対象になる。
        roll, pitch, yaw = rng.standard_normal(3) * 0.15
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        quat = np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])
        quat = quat / np.linalg.norm(quat)
        ang_vel = rng.standard_normal(3) * 0.5
        # 法線力 [N]。閾値 1.0 N の前後を含める（0 / 0.5 / 数十 N）。
        foot_force = rng.choice([0.0, 0.5, 1.0, 5.0, 40.0], size=2).astype(float)
        command = commands[index % len(commands)]
        previous_action = rng.standard_normal(JOINT_NUM) * 0.3
        current_action = rng.standard_normal(JOINT_NUM) * 0.3
        joint_position = rng.standard_normal(JOINT_NUM) * 0.2
        joint_velocity = rng.standard_normal(JOINT_NUM) * 1.0
        row = [float(step_count), base_height]
        row += [float(v) for v in quat]
        row += [float(v) for v in ang_vel]
        row += [float(v) for v in foot_force]
        row += [float(v) for v in command]
        row += [float(v) for v in previous_action]
        row += [float(v) for v in current_action]
        row += [float(v) for v in joint_position]
        row += [float(v) for v in joint_velocity]
        cases.append(row)
    return cases


def run(command: list[str], stdin_text: str, label: str) -> np.ndarray:
    result = subprocess.run(command, input=stdin_text, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FATAL] {label} が失敗 (exit={result.returncode})", file=sys.stderr)
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(1)
    reader = csv.reader(io.StringIO(result.stdout))
    header = next(reader)
    rows = [[float(v) for v in row] for row in reader if row]
    return np.array(rows, dtype=np.float64), header


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isaac-root", type=Path, default=DEFAULT_ISAAC_ROOT)
    parser.add_argument("--isaac-python", type=Path, default=Path("/home/pomiou/work/env_isaaclab/bin/python"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--dump-binary", type=Path, default=REPO_ROOT / "build-linux" / "dump_policy_io")
    parser.add_argument("--cases", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.dump_binary.is_file():
        raise SystemExit(f"dump_policy_io が無い: {args.dump_binary}（cmake --build build-linux）")

    cases = build_cases(args.cases, args.seed)
    stdin_text = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in cases) + "\n"

    cpp, cpp_header = run([str(args.dump_binary)], stdin_text, "C++ dump_policy_io")
    isaac, isaac_header = run(
        [
            str(args.isaac_python),
            str(args.isaac_root / "scripts" / "dump_big_rabbit_policy_io.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--params",
            str(args.params),
        ],
        stdin_text,
        "Isaac dump_big_rabbit_policy_io.py",
    )

    if cpp_header != isaac_header:
        print("[FATAL] 列名が違う", file=sys.stderr)
        print(f"  cpp  : {cpp_header[:6]} ... ({len(cpp_header)} 列)", file=sys.stderr)
        print(f"  isaac: {isaac_header[:6]} ... ({len(isaac_header)} 列)", file=sys.stderr)
        return 1
    if cpp.shape != isaac.shape:
        print(f"[FATAL] 形が違う cpp={cpp.shape} isaac={isaac.shape}", file=sys.stderr)
        return 1

    diff = np.abs(cpp - isaac)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(cpp_header):
        prefix = name.rstrip("0123456789") or name
        groups.setdefault(prefix, []).append(index)

    print(f"ケース数 {cpp.shape[0]} / 列数 {cpp.shape[1]}")
    print(f"{'group':<12}{'列数':>6}{'最大差':>14}{'許容':>12}  判定   最悪列")
    print("-" * 74)
    failures = 0
    for prefix, indices in groups.items():
        block = diff[:, indices]
        worst = float(block.max())
        worst_column = indices[int(np.unravel_index(block.argmax(), block.shape)[1])]
        tolerance = TOLERANCE.get(prefix, 1.0e-5)
        ok = worst <= tolerance
        if not ok:
            failures += 1
        print(
            f"{prefix:<12}{len(indices):>6}{worst:>14.3e}{tolerance:>12.0e}"
            f"  {'OK  ' if ok else 'FAIL'}   {cpp_header[worst_column]}"
        )

    if failures:
        # どのケースでずれたかを見せる。位相依存の不具合は特定位相でしか出ない。
        print("\n差が大きいケース（上位 5 件）:")
        per_case = diff.max(axis=1)
        for case_index in np.argsort(per_case)[::-1][:5]:
            column = int(diff[case_index].argmax())
            print(
                f"  case {case_index:>4}  phase={cpp[case_index, 0]:.6f}"
                f"  列 {cpp_header[column]:<12} cpp={cpp[case_index, column]:+.9g}"
                f" isaac={isaac[case_index, column]:+.9g}  差={diff[case_index, column]:.3e}"
            )
        print(f"\nFAILED ({failures} groups)")
        return 1

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
