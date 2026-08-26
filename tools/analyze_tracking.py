#!/usr/bin/env python3
"""追従性能の trace CSV を解析し、Isaac 側の基準値と突き合わせる。

big_rabbit_isaac/scripts/check_big_rabbit_reference_tracking.py と同じ量を出す。
あちらは骨盤を空中に溶接した MuJoCo で「関節側 PD」を直接掛けたもの、
こちらは big_rabbit_mujoco が「符号 x 減速比 -> モータ PD -> 減速機でトルクを戻す」経路で
出したもの。両者が一致すれば、ドライバ経路が関節側 PD と等価だと言える。

    BIG_RABBIT_SCENE_XML=robotmodel/big_rabbit/scene_tracking.xml \
    BIG_RABBIT_CONTROL_MODE=reference BIG_RABBIT_HEADLESS=1 \
    BIG_RABBIT_MAX_SIM_TIME=6 BIG_RABBIT_RL_START_S=0 BIG_RABBIT_LOG_INTERVAL_S=0 \
    BIG_RABBIT_MOTION_COMMAND_X=1 BIG_RABBIT_TRACE_CSV=/tmp/trace.csv \
      ./build-linux/big_rabbit_mujoco_sim
    python3 tools/analyze_tracking.py /tmp/trace.csv --period 0.7
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


# check_big_rabbit_reference_tracking.py が同条件で出した値。
#   参照 walk_ref_s140_l040.npz / 周期 0.7 s / ゲイン x1 / armature 0.01
#   物理 1 kHz / 目標 62.5 Hz 保持（Isaac と実機経路に合わせる）
# あちらの条件を揃えるまでに 2 つ直した。
#   - MJCF の timestep 0.002 をそのまま使っていた（Isaac は 1 kHz）
#   - data.ctrl に関節側トルクを書いていた（MuJoCo は tau = gear * ctrl なので gear 倍過大）
# 足裏カプセル化（v22）でも再測定して同値を確認済み。骨盤を空中に溶接するので
# 足裏形状には依存しない測定。
ISAAC_BASELINE = {
    "left_hip_pitch_joint": {"ratio": 0.809, "lag_deg": 26.7},
    "left_knee_joint": {"ratio": 0.857, "lag_deg": 24.2},
}
RATIO_TOLERANCE = 0.02
LAG_TOLERANCE_DEG = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--period", type=float, default=0.7)
    parser.add_argument("--settle-cycles", type=float, default=1.0, help="捨てる先頭の周期数")
    args = parser.parse_args()

    with args.trace.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader if row], dtype=np.float64)

    time = rows[:, 0]
    names = [name[4:] for name in header if name.startswith("cmd_")]
    joint_num = len(names)
    cmd = rows[:, 1 : 1 + joint_num]
    act = rows[:, 1 + joint_num : 1 + 2 * joint_num]
    tau_ratio = rows[:, 1 + 2 * joint_num]

    # 先頭の過渡を捨てる。
    keep = time >= time[0] + args.settle_cycles * args.period
    time, cmd, act, tau_ratio = time[keep], cmd[keep], act[keep], tau_ratio[keep]
    dt = float(np.median(np.diff(time)))

    print(f"サンプル {len(time)} 点 / dt {dt * 1000:.3f} ms / 周期 {args.period} s")
    print(f"{'joint':<24}{'指令振幅':>10}{'実測振幅':>10}{'振幅比':>9}{'位相遅れ':>10}")
    print("-" * 65)
    results = {}
    for index, name in enumerate(names):
        c = cmd[:, index]
        a = act[:, index]
        cmd_amp = float(c.max() - c.min())
        act_amp = float(a.max() - a.min())
        if cmd_amp < 1.0e-6:
            print(f"{name:<24}{np.degrees(cmd_amp):>10.3f}{np.degrees(act_amp):>10.3f}{'--':>9}{'--':>10}")
            continue
        # 位相遅れは相互相関のピーク位置から。Python 側の解析と同じ方法。
        corr = np.correlate(a - a.mean(), c - c.mean(), mode="full")
        lag_steps = int(np.argmax(corr)) - (len(c) - 1)
        lag_deg = 360.0 * (lag_steps * dt) / args.period
        ratio = act_amp / cmd_amp
        results[name] = {"ratio": ratio, "lag_deg": lag_deg}
        print(
            f"{name:<24}{np.degrees(cmd_amp):>10.3f}{np.degrees(act_amp):>10.3f}"
            f"{ratio:>9.3f}{lag_deg:>9.1f}°"
        )

    print(f"\nトルク使用率 最大 {tau_ratio.max():.3f} / 平均 {tau_ratio.mean():.3f}")

    print("\nIsaac 側の基準値との比較（関節側 PD 直掛け vs ドライバ経路）")
    print(f"{'joint':<24}{'基準比':>9}{'実測比':>9}{'差':>9}{'基準遅れ':>10}{'実測遅れ':>10}{'差':>8}  判定")
    print("-" * 88)
    failures = 0
    for name, baseline in ISAAC_BASELINE.items():
        if name not in results:
            print(f"{name:<24}  (trace に無い)")
            failures += 1
            continue
        got = results[name]
        ratio_diff = abs(got["ratio"] - baseline["ratio"])
        lag_diff = abs(got["lag_deg"] - baseline["lag_deg"])
        ok = ratio_diff <= RATIO_TOLERANCE and lag_diff <= LAG_TOLERANCE_DEG
        if not ok:
            failures += 1
        print(
            f"{name:<24}{baseline['ratio']:>9.3f}{got['ratio']:>9.3f}{ratio_diff:>9.3f}"
            f"{baseline['lag_deg']:>9.1f}°{got['lag_deg']:>9.1f}°{lag_diff:>7.1f}°  {'OK' if ok else 'FAIL'}"
        )

    print(f"\n{'ALL PASS' if failures == 0 else f'FAILED ({failures})'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
