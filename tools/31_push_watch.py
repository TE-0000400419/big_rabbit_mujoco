#!/usr/bin/env python3
"""押されたときの応答を GUI で見る。30_push_test.py の 1 条件版。

30 は headless で回して合否だけを出す。こちらは同じ押し板を同じ式で置いたまま、
**画面付きで sim を起動**して、押される瞬間と踏ん張りをその場で見るためのもの。
制御本体（src / dep）には手を入れない。押し板の理屈は 30 のコメントを見る。

    # 立ち姿勢を正面から 4 N s で押すのを見る（既定）
    python3 tools/31_push_watch.py

    # 前進中に左から 6 N s。実時間で見たあと 0.25 倍速で見直す
    python3 tools/31_push_watch.py --command-x 1 --dir left --impulse 6 --replay --speed 0.25

    # ゲームパッドで歩かせながら押す（別端末で tools/gamepad_command.py）
    python3 tools/31_push_watch.py --command-source udp --impulse 6

GUI は C++ sim 側のビューア。マウスで視点、ウィンドウを閉じれば終了。
実時間なので衝突自体は一瞬（0.1 s 程度）で終わる。**コマ送りで見たいときは
--replay**。記録した qpos を Python 側のビューアで好きな速度で流し直せる
（描き戻すだけなので、見えるものは実行結果そのもの）。

終了後、30 と同じ判定（転倒 / 復帰、実測力積、最大傾き、最低骨盤高）を出し、
応答の時系列を outputs/push_watch_<条件>.png に描く（--no-plot で止める）。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"


def load_push_test():
    """30_push_test.py を module として読む。押し板の式と判定を二重に持たないため。"""
    path = REPO_ROOT / "tools" / "30_push_test.py"
    spec = importlib.util.spec_from_file_location("push_test", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"読めない: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass の解決に sys.modules 登録が要る（importlib 直読みのときの定番）。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PT = load_push_test()


def run_gui(sim: Path, scene: Path, qpos_csv: Path, seconds: float, command_x: float,
            command_yaw: float, command_source: str, qpos_hz: float, rl_start: float) -> None:
    """画面付きで sim を起動する。BIG_RABBIT_HEADLESS を渡さないのがここの肝。"""
    env = dict(os.environ)
    env.update(
        BIG_RABBIT_SCENE_XML=str(scene),
        BIG_RABBIT_LOG_INTERVAL_S="1",
        BIG_RABBIT_MOTION_COMMAND_X=f"{command_x}",
        BIG_RABBIT_MOTION_COMMAND_YAW=f"{command_yaw}",
        BIG_RABBIT_COMMAND_SOURCE=command_source,
        BIG_RABBIT_QPOS_CSV=str(qpos_csv),
        BIG_RABBIT_QPOS_HZ=f"{qpos_hz}",
        BIG_RABBIT_RL_START_S=f"{rl_start}",
    )
    env.pop("BIG_RABBIT_HEADLESS", None)
    if seconds > 0.0:
        env["BIG_RABBIT_MAX_SIM_TIME"] = f"{seconds}"
    else:
        env.pop("BIG_RABBIT_MAX_SIM_TIME", None)
    subprocess.run([str(sim)], env=env, check=True)


def replay(scene: Path, qpos_csv: Path, speed: float, start: float, end: float) -> None:
    """記録した qpos を Python 側のビューアで流し直す。物理は回さない。"""
    import mujoco
    import mujoco.viewer

    time_column, qpos = PT.load_qpos(qpos_csv)
    window = (time_column >= start) & (time_column <= end) if end > start else \
        np.ones_like(time_column, dtype=bool)
    time_column, qpos = time_column[window], qpos[window]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if qpos.shape[1] != model.nq:
        raise SystemExit(f"nq 不一致: CSV {qpos.shape[1]} / model {model.nq}")

    print(f"[replay] {time_column[0]:.2f}-{time_column[-1]:.2f} s を {speed:g} 倍速でループ。"
          f"ウィンドウを閉じると終了")
    dt = float(np.median(np.diff(time_column)))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for index in range(len(time_column)):
                if not viewer.is_running():
                    break
                data.qpos[:] = qpos[index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt / max(speed, 1.0e-3))
            time.sleep(0.5)


def plot_response(qpos_csv: Path, plate_adr: int, push_dir: np.ndarray, t_hit: float,
                  out: Path, title: str) -> None:
    """骨盤高 / 傾き / 押し方向速度 / 板の位置を 1 枚に。ラベルは ASCII（日本語フォント無し）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_column, qpos = PT.load_qpos(qpos_csv)
    tilt = PT.quat_tilt_deg(qpos[:, 3:7])
    velocity = np.gradient(qpos[:, 0:2], time_column, axis=0) @ push_dir

    figure, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(time_column, qpos[:, 2], lw=1.2)
    axes[0].set_ylabel("pelvis z [m]")
    axes[1].plot(time_column, tilt, lw=1.2, color="tab:orange")
    axes[1].set_ylabel("tilt [deg]")
    axes[2].plot(time_column, velocity, lw=1.2, color="tab:green")
    axes[2].set_ylabel("base vel along push [m/s]")
    axes[3].plot(time_column, qpos[:, plate_adr], lw=1.2, color="tab:red")
    axes[3].set_ylabel("plate travel [m]")
    axes[3].set_xlabel("sim time [s]")
    for axis in axes:
        axis.axvline(t_hit, color="k", ls="--", lw=0.8)
        axis.grid(alpha=0.3)
    axes[0].set_title(title)
    figure.tight_layout()
    out.parent.mkdir(exist_ok=True)
    figure.savefig(out, dpi=110)
    plt.close(figure)
    print(f"[plot] {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, default=PT.DEFAULT_SIM)
    parser.add_argument("--scene", type=Path, default=PT.DEFAULT_SCENE)
    parser.add_argument("--dir", default="front", help="板が来る方向 front/back/left/right")
    parser.add_argument("--impulse", type=float, default=4.0, help="公称力積 [N s]")
    parser.add_argument("--force", type=float, default=None, help="力 [N]（--duration と組）")
    parser.add_argument("--duration", type=float, default=0.1, help="--force の作用時間 [s]")
    parser.add_argument("--command-x", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-source", default="env", choices=["env", "udp", "sequence"],
                        help="udp はゲームパッド（tools/gamepad_command.py）")
    parser.add_argument("--t-hit", type=float, default=4.0, help="押しをぶつける時刻 [s]")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="sim 時間 [s]。0 ならウィンドウを閉じるまで")
    parser.add_argument("--rl-start", type=float, default=1.0)
    parser.add_argument("--pusher-mass", type=float, default=3.0)
    parser.add_argument("--plate-half", default="0.010,0.180,0.100")
    parser.add_argument("--fall-height", type=float, default=0.28)
    parser.add_argument("--fall-tilt", type=float, default=45.0)
    parser.add_argument("--qpos-hz", type=float, default=200.0)
    parser.add_argument("--replay", action="store_true", help="実行後に記録を流し直す")
    parser.add_argument("--speed", type=float, default=0.25, help="--replay の再生速度")
    parser.add_argument("--replay-window", type=float, default=3.0,
                        help="--replay で見る範囲 [s]（押しの前後）。0 で全区間")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--keep-scene", action="store_true")
    args = parser.parse_args()

    if not args.sim.exists():
        raise SystemExit(f"sim が無い: {args.sim}（make を先に走らせる）")
    if args.dir not in PT.APPROACH_BODY:
        raise SystemExit(f"--dir が不正: {args.dir}（front/back/left/right）")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit("DISPLAY が無い。画面付きで動かせないので 30_push_test.py を使う")
    if args.t_hit <= args.rl_start + 0.5:
        raise SystemExit("--t-hit は RL 開始より十分あとにする")
    half = tuple(float(value) for value in args.plate_half.split(","))
    if len(half) != 3:
        raise SystemExit("--plate-half は 厚み,幅,高さ の 3 つ")
    impulse = args.force * args.duration if args.force is not None else args.impulse

    import mujoco

    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = f"{args.dir}_{impulse:g}"
    scene = args.scene.parent / f"{PT.SCENE_PREFIX}watch_{tag}.xml"
    qpos_csv = OUTPUT_DIR / f"push_watch_{tag}.csv"
    baseline_csv = OUTPUT_DIR / "push_watch_baseline.csv"

    # ---- 狙いを決めるための下見。ここだけは headless で先に回す ----
    # 指令が env なら押しまでの軌跡は決定論的なので、baseline の t_hit 断面が
    # そのまま本番の姿勢になる。ゲームパッド等では毎回変わるので、初期位置を狙う。
    aiming_is_exact = args.command_source == "env"
    baseline_seconds = max(args.t_hit + 0.5, 2.0) if aiming_is_exact else max(args.rl_start + 1.0, 2.0)
    print(f"[下見] headless で {baseline_seconds:.1f} s だけ回して狙いを決める")
    PT.run_sim(args.sim, args.scene, baseline_csv, baseline_seconds,
               args.command_x, args.command_yaw, args.qpos_hz, args.rl_start)
    baseline_t, baseline_q = PT.load_qpos(baseline_csv)
    hit_index = int(np.argmin(np.abs(baseline_t - args.t_hit))) if aiming_is_exact else -1
    pelvis_xy = baseline_q[hit_index, 0:2].copy()
    pelvis_z = float(baseline_q[hit_index, 2])
    yaw = PT.quat_yaw(baseline_q[hit_index, 3:7])
    if not aiming_is_exact:
        print("[下見] 指令が env でないので、狙いは初期位置に固定する。"
              "押しの時刻にロボットがそこから離れていると空振りする")

    body_dir = PT.APPROACH_BODY[args.dir]
    approach = np.array([
        body_dir[0] * math.cos(yaw) - body_dir[1] * math.sin(yaw),
        body_dir[0] * math.sin(yaw) + body_dir[1] * math.cos(yaw),
    ])
    reach = PT.robot_reach(args.scene, baseline_q[hit_index], pelvis_xy, approach,
                           pelvis_z - half[2], pelvis_z + half[2])

    speed = impulse / args.pusher_mass
    omega = math.pi / (2.0 * args.t_hit)
    travel = speed / omega
    stiffness = args.pusher_mass * omega * omega
    plate_xy = pelvis_xy + approach * (reach + half[0] + travel + 0.005)
    PT.write_push_scene(scene, args.scene, tag,
                        np.array([plate_xy[0], plate_xy[1], pelvis_z]),
                        math.atan2(-approach[1], -approach[0]),
                        -approach, travel, stiffness, args.pusher_mass, half)

    try:
        print(f"[GUI] {args.dir} から p={impulse:g} N s（板 {args.pusher_mass:g} kg を "
              f"{speed:.2f} m/s、助走 {travel:.2f} m）。t={args.t_hit:g} s に当たる")
        print("      マウスで視点。ウィンドウを閉じると終了"
              + ("" if args.seconds <= 0 else f"（{args.seconds:g} s で自動終了）"))
        run_gui(args.sim, scene, qpos_csv, args.seconds, args.command_x, args.command_yaw,
                args.command_source, args.qpos_hz, args.rl_start)

        model = mujoco.MjModel.from_xml_path(str(scene))
        plate_adr = int(model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_slide")])
        condition = PT.Condition(args.dir, impulse, speed, travel, stiffness, scene, qpos_csv)
        result = PT.analyze(condition, baseline_t, baseline_q, plate_adr, args.pusher_mass,
                            args.t_hit, args.fall_height, args.fall_tilt, approach, {})

        verdict = "転倒" if result.fell else ("復帰" if result.recovered else "耐えた(未収束)")
        # 図のタイトルは ASCII にする（matplotlib の既定フォントに日本語が無い）。
        verdict_ascii = "FELL" if result.fell else ("recovered" if result.recovered else "survived")
        if not result.hit:
            verdict = "空振り"
            verdict_ascii = "MISS"
        print()
        print(f"  判定        : {verdict}")
        print(f"  実測の力積  : {result.delivered:.2f} N s（公称 {impulse:g}）")
        print(f"  骨盤の Δv   : {result.base_dv:+.3f} m/s（押し方向）")
        print(f"  最大傾き    : {result.max_tilt_deg:.2f} deg")
        print(f"  最低骨盤高  : {result.min_pelvis_z:.4f} m")
        if aiming_is_exact:
            print(f"  押し無しとの位置差: {result.drift_m:.3f} m")

        if not args.no_plot:
            plot_response(qpos_csv, plate_adr, -approach, args.t_hit,
                          OUTPUT_DIR / f"push_watch_{tag}.png",
                          f"push {args.dir} {impulse:g} Ns  (cmd x={args.command_x:g}, "
                          f"yaw={args.command_yaw:g})  ->  {verdict_ascii}")
        if args.replay:
            window = args.replay_window
            replay(scene, qpos_csv, args.speed,
                   args.t_hit - window if window > 0 else 0.0,
                   args.t_hit + window if window > 0 else 0.0)
    finally:
        if not args.keep_scene:
            scene.unlink(missing_ok=True)

    return 1 if result.fell else 0


if __name__ == "__main__":
    raise SystemExit(main())
