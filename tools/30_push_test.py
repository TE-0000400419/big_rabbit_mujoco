#!/usr/bin/env python3
"""外乱（押し）に対する転倒耐性を測る。段 6 の次に置く想定の検証。

**制御本体（src / dep）には一切手を入れない。**
やっていることは 3 つだけ:

  1. scene.xml を include した「押し板つきシーン」を生成する
  2. 既存の C++ sim をそのまま headless で回す（物理も制御も向こうが正本）
  3. 吐かせた qpos CSV を読んで、押された後に転倒したかを判定する

押し板は水平 slide 関節にバネ（stiffness / springref）を付けただけの板。
MuJoCo は XML で「時刻 t に力を加える」が書けないので、バネで板を助走させ、
**最高速に達する位置がちょうどロボットの表面**になるよう置く。これで
アクチュエータも C++ の外力注入も無しに、狙った時刻に狙った運動量をぶつけられる。

    q(t) = D (1 - cos wt),  v(t) = D w sin wt,  w = pi / (2 t_hit)
    衝突時刻 t_hit で v = D w が最大、公称力積 p = m_plate * D w
    -> w = pi / (2 t_hit),  D = (p / m_plate) / w,  k = m_plate w^2

板は衝突後バネに引き戻されるので、押しは 1 発のパルスになる。

接触フィルタ: 板は contype=2 / conaffinity=2。
胴・脚（2/1）とは当たり、床（1/1）とも足裏（4/1）とも当たらない。
足裏力センサに押し板の力が混入しないので、bridge の接地判定は汚れない。

使い方:

    # 立ち姿勢で前後左右から押す（既定の力積スイープ）
    python3 tools/30_push_test.py

    # 前進歩行中に横から押す。転ぶまで力積を上げる
    python3 tools/30_push_test.py --command-x 1 --dir left,right --impulse 2,4,6,8

    # 動画も出す（videos/push_*.mp4）
    python3 tools/30_push_test.py --dir front --impulse 6 --video

判定は qpos だけを使う。sim が吐く [EVAL] 行も拾って表に載せる。
押される様子を画面で見たいときは tools/31_push_watch.py（同じ押し板を GUI で回す）。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM = REPO_ROOT / "build-linux" / "big_rabbit_mujoco_sim"
DEFAULT_SCENE = REPO_ROOT / "robotmodel" / "big_rabbit" / "scene.xml"
OUTPUT_DIR = REPO_ROOT / "outputs"
VIDEO_DIR = REPO_ROOT / "videos"

# 生成シーンは include の相対パス解決の都合でモデルと同じ階層に置くしかない。
# `_push_` 始まりのものは全部この script の生成物（既定では実行後に消す）。
SCENE_PREFIX = "_push_"

# 「板が来る方向」-> ロボット体幹座標で板が待ち構える向き。
# front なら +x 側（正面）に板があり、力は -x（後ろ向き）に働く。
APPROACH_BODY = {
    "front": (+1.0, 0.0),
    "back": (-1.0, 0.0),
    "left": (0.0, +1.0),
    "right": (0.0, -1.0),
}


@dataclass
class Condition:
    direction: str
    impulse: float          # 公称力積 [N s]
    speed: float            # 衝突時の板の速度 [m/s]
    travel: float           # 助走距離 D [m]
    stiffness: float        # バネ定数 k [N/m]
    scene: Path
    qpos_csv: Path


@dataclass
class Result:
    condition: Condition
    delivered: float        # 実測の力積（板の運動量変化）[N s]
    base_dv: float          # 骨盤の速度変化（押し方向成分）[m/s]
    max_tilt_deg: float
    min_pelvis_z: float
    drift_m: float          # 押し後の水平変位（baseline との差）[m]
    fell: bool
    recovered: bool
    hit: bool
    tau_ratio_max: float


# ---------------------------------------------------------------- sim を回す


def run_sim(sim: Path, scene: Path, qpos_csv: Path, seconds: float, command_x: float,
            command_yaw: float, qpos_hz: float, rl_start: float) -> dict[str, str]:
    """C++ sim を headless で 1 本回して [EVAL] を辞書で返す。"""
    env = dict(os.environ)
    env.update(
        BIG_RABBIT_SCENE_XML=str(scene),
        BIG_RABBIT_HEADLESS="1",
        BIG_RABBIT_MAX_SIM_TIME=f"{seconds}",
        BIG_RABBIT_LOG_INTERVAL_S="0",
        BIG_RABBIT_MOTION_COMMAND_X=f"{command_x}",
        BIG_RABBIT_MOTION_COMMAND_YAW=f"{command_yaw}",
        BIG_RABBIT_QPOS_CSV=str(qpos_csv),
        BIG_RABBIT_QPOS_HZ=f"{qpos_hz}",
        BIG_RABBIT_RL_START_S=f"{rl_start}",
    )
    completed = subprocess.run([str(sim)], env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout + completed.stderr)
        raise SystemExit(f"sim が異常終了した（scene={scene.name}）")
    evals: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        matched = re.match(r"\[EVAL\]\s+(\S+)\s+=\s+(.*)", line)
        if matched:
            evals[matched.group(1)] = matched.group(2).strip()
    return evals


def load_qpos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise SystemExit(f"qpos CSV が短すぎる: {path}")
    return rows[:, 0], rows[:, 1:]


# ------------------------------------------------------------ 幾何・姿勢の計算


def quat_tilt_deg(quat: np.ndarray) -> np.ndarray:
    """(w,x,y,z) の列から、体幹 z 軸と鉛直のなす角 [deg] を返す。"""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt))


def quat_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def robot_reach(scene: Path, qpos_row: np.ndarray, pelvis_xy: np.ndarray,
                approach: np.ndarray, z_lo: float, z_hi: float) -> float:
    """t_hit の姿勢で、骨盤中心から板の側へロボットが張り出している量 [m]。

    板の高さ帯（z_lo..z_hi）に掛かる衝突 geom だけを見て、境界球で外側を取る。
    脚が振り出されていても板が空振りしないよう、姿勢そのものから測る。
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if qpos_row.size != model.nq:
        raise SystemExit(f"baseline の nq 不一致: CSV {qpos_row.size} / model {model.nq}")
    data.qpos[:] = qpos_row
    mujoco.mj_forward(model, data)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    reach = -1.0e9
    for geom_id in range(model.ngeom):
        if geom_id == floor_id:
            continue
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        center = data.geom_xpos[geom_id]
        radius = float(model.geom_rbound[geom_id])
        if center[2] + radius < z_lo or center[2] - radius > z_hi:
            continue
        reach = max(reach, float((center[:2] - pelvis_xy) @ approach) + radius)
    if reach < -1.0e8:
        raise SystemExit("板の高さ帯に衝突 geom が無い。--plate-z-half を広げる")
    return reach


# ---------------------------------------------------------------- シーン生成


def write_push_scene(path: Path, base_scene: Path, condition_name: str, plate_pos: np.ndarray,
                     yaw: float, axis_world: np.ndarray, travel: float, stiffness: float,
                     mass: float, half: tuple[float, float, float]) -> None:
    """scene.xml を include して押し板だけを足したシーンを書く。

    slide の軸は「力の向き」。板は q=0 から出発して travel だけ進み、
    ちょうどそこで最高速になる（そこにロボットが居るように置いてある）。
    inertiafromgeom=false のモデルなので inertial は明示する。
    """
    thickness, width, height = half
    inertia = (
        mass * (width ** 2 + height ** 2) / 3.0,
        mass * (thickness ** 2 + height ** 2) / 3.0,
        mass * (thickness ** 2 + width ** 2) / 3.0,
    )
    path.write_text(
        f'<mujoco model="big_rabbit push scene ({condition_name})">\n'
        f'  <!-- tools/30_push_test.py の生成物。手で編集しない。 -->\n'
        f'  <include file="{base_scene.name}"/>\n'
        f'  <worldbody>\n'
        f'    <body name="pusher" pos="{plate_pos[0]:.6f} {plate_pos[1]:.6f} {plate_pos[2]:.6f}"'
        f' euler="0 0 {yaw:.6f}">\n'
        f'      <inertial pos="0 0 0" mass="{mass:.6f}"'
        f' diaginertia="{inertia[0]:.6f} {inertia[1]:.6f} {inertia[2]:.6f}"/>\n'
        f'      <joint name="pusher_slide" type="slide" axis="1 0 0"'
        f' stiffness="{stiffness:.6f}" springref="{travel:.6f}" damping="0"/>\n'
        f'      <geom name="pusher_plate" type="box"'
        f' size="{thickness:.4f} {width:.4f} {height:.4f}"\n'
        f'            contype="2" conaffinity="2" group="2" rgba="0.90 0.25 0.20 0.9"/>\n'
        f'    </body>\n'
        f'  </worldbody>\n'
        f'</mujoco>\n',
        encoding="utf-8",
    )
    _ = axis_world  # 軸は body の yaw で表現しているので XML には出さない


# ------------------------------------------------------------------ 判定


def analyze(condition: Condition, baseline_t: np.ndarray, baseline_q: np.ndarray,
            plate_adr: int, mass: float, t_hit: float, fall_height: float,
            fall_tilt_deg: float, approach_world: np.ndarray, evals: dict[str, str]) -> Result:
    time, qpos = load_qpos(condition.qpos_csv)
    dt = float(np.median(np.diff(time)))
    push_dir = -approach_world  # 力の向き（板 -> ロボット）

    plate_q = qpos[:, plate_adr]
    plate_v = np.gradient(plate_q, time)

    # 衝突時刻 = 板が最も急減速した瞬間。バネ由来の自然な減速はこれよりずっと緩い。
    window = (time > t_hit - 1.0) & (time < t_hit + 1.0)
    plate_a = np.gradient(plate_v, time)
    impact_index = int(np.argmin(np.where(window, plate_a, 0.0)))
    t_impact = float(time[impact_index])

    def mean_in(values: np.ndarray, lo: float, hi: float) -> float:
        mask = (time >= lo) & (time <= hi)
        return float(values[mask].mean()) if mask.any() else float("nan")

    v_pre = mean_in(plate_v, t_impact - 0.15, t_impact - 0.02)
    v_post = mean_in(plate_v, t_impact + 0.03, t_impact + 0.15)
    delivered = mass * (v_pre - v_post)
    hit = delivered > 0.05 * condition.impulse

    # 骨盤の押し方向速度。50-200 Hz の差分なので平滑化してから前後を比べる。
    pelvis_xy = qpos[:, 0:2]
    pelvis_v = np.gradient(pelvis_xy, time, axis=0) @ push_dir
    base_dv = mean_in(pelvis_v, t_impact + 0.02, t_impact + 0.12) - \
        mean_in(pelvis_v, t_impact - 0.15, t_impact - 0.02)

    after = time >= t_impact - 0.05
    tilt = quat_tilt_deg(qpos[:, 3:7])
    max_tilt = float(tilt[after].max())
    min_z = float(qpos[after, 2].min())
    fell = bool((qpos[after, 2] < fall_height).any() or (tilt[after] > fall_tilt_deg).any())

    # 押しが無ければどこに居たか（baseline）との差。転ばなくても流されたかを見る。
    end_t = float(time[-1])
    base_index = int(np.argmin(np.abs(baseline_t - end_t)))
    drift = float(np.linalg.norm(qpos[-1, 0:2] - baseline_q[base_index, 0:2]))

    recovered = bool(
        not fell
        and qpos[-1, 2] > fall_height + 0.10
        and tilt[-1] < 15.0
        and float(np.abs(pelvis_v[-int(1.0 / dt):]).mean()) < 1.0
    )

    return Result(
        condition=condition,
        delivered=float(delivered),
        base_dv=float(base_dv),
        max_tilt_deg=max_tilt,
        min_pelvis_z=min_z,
        drift_m=drift,
        fell=fell,
        recovered=recovered,
        hit=bool(hit),
        tau_ratio_max=float(evals.get("tau_ratio_max", "nan")),
    )


# ------------------------------------------------------------------ 動画


def render_video(qpos_csv: Path, scene: Path, out: Path, fps: int, qpos_hz: float) -> None:
    """解析用の高レート CSV を間引いてから render_qpos_video.py に渡す。"""
    time, qpos = load_qpos(qpos_csv)
    step = max(1, int(round(qpos_hz / fps)))
    thinned = qpos_csv.with_name(qpos_csv.stem + f"_{fps}hz.csv")
    header = qpos_csv.read_text(encoding="utf-8").splitlines()[0]
    rows = np.column_stack([time, qpos])[::step]
    np.savetxt(thinned, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "render_qpos_video.py"), str(thinned),
         "--out", str(out), "--scene", str(scene), "--fps", str(fps)],
        check=True,
    )


# ------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--dir", default="front,back,left,right",
                        help="板が来る方向（体幹基準）。front,back,left,right から選ぶ")
    parser.add_argument("--impulse", default="2,4,6,8",
                        help="公称力積 [N s] のリスト。小さい順に試す")
    parser.add_argument("--force", type=float, default=None,
                        help="力 [N] で指定する場合。--duration と組で力積に直す")
    parser.add_argument("--duration", type=float, default=0.1, help="--force の作用時間 [s]")
    parser.add_argument("--command-x", type=float, default=0.0, help="移動指令 x（0 なら立ち）")
    parser.add_argument("--command-yaw", type=float, default=0.0, help="移動指令 yaw")
    parser.add_argument("--t-hit", type=float, default=4.0, help="押しをぶつける時刻 [s]")
    parser.add_argument("--seconds", type=float, default=9.0, help="1 本あたりの sim 時間 [s]")
    parser.add_argument("--rl-start", type=float, default=1.0, help="RL に切り替える時刻 [s]")
    parser.add_argument("--pusher-mass", type=float, default=3.0, help="押し板の質量 [kg]")
    parser.add_argument("--plate-half", default="0.010,0.180,0.100",
                        help="板の半寸法 厚み,幅,高さ [m]")
    parser.add_argument("--fall-height", type=float, default=0.28, help="転倒とみなす骨盤高 [m]")
    parser.add_argument("--fall-tilt", type=float, default=45.0, help="転倒とみなす傾き [deg]")
    parser.add_argument("--qpos-hz", type=float, default=200.0, help="qpos CSV のレート")
    parser.add_argument("--continue-after-fall", action="store_true",
                        help="転倒しても同じ方向でより大きい力積を試す")
    parser.add_argument("--video", action="store_true", help="条件ごとに mp4 を書き出す")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--keep-scene", action="store_true", help="生成シーンを消さない")
    parser.add_argument("--csv", type=Path, default=OUTPUT_DIR / "push_test_summary.csv")
    args = parser.parse_args()

    if not args.sim.exists():
        raise SystemExit(f"sim が無い: {args.sim}（make を先に走らせる）")
    directions = [name.strip() for name in args.dir.split(",") if name.strip()]
    for name in directions:
        if name not in APPROACH_BODY:
            raise SystemExit(f"--dir が不正: {name}（front/back/left/right）")
    if args.force is not None:
        impulses = [args.force * args.duration]
    else:
        impulses = sorted(float(value) for value in args.impulse.split(",") if value.strip())
    half = tuple(float(value) for value in args.plate_half.split(","))
    if len(half) != 3:
        raise SystemExit("--plate-half は 厚み,幅,高さ の 3 つ")
    if args.t_hit <= args.rl_start + 0.5:
        raise SystemExit("--t-hit は RL 開始より十分あとにする")
    if args.t_hit + 2.0 > args.seconds:
        raise SystemExit("--seconds が短い。押しの後 2 s 以上は見る")

    OUTPUT_DIR.mkdir(exist_ok=True)
    scene_dir = args.scene.parent
    generated: list[Path] = []

    # ---- baseline。押し無しで同じ指令を回し、狙う姿勢と「押されなかった場合」を得る ----
    baseline_csv = OUTPUT_DIR / "push_baseline.csv"
    print(f"[baseline] command=({args.command_x}, {args.command_yaw})  {args.seconds} s")
    baseline_evals = run_sim(args.sim, args.scene, baseline_csv, args.seconds,
                             args.command_x, args.command_yaw, args.qpos_hz, args.rl_start)
    baseline_t, baseline_q = load_qpos(baseline_csv)
    hit_index = int(np.argmin(np.abs(baseline_t - args.t_hit)))
    pelvis_xy = baseline_q[hit_index, 0:2].copy()
    pelvis_z = float(baseline_q[hit_index, 2])
    yaw = quat_yaw(baseline_q[hit_index, 3:7])
    # 比較は「押した後の窓」と揃える。起動直後の crouch 沈み込みは入れない。
    baseline_after = baseline_t >= args.t_hit - 0.05
    baseline_tilt = quat_tilt_deg(baseline_q[:, 3:7])[baseline_after]
    if float(baseline_q[:, 2].min()) < args.fall_height:
        raise SystemExit("baseline の時点で転んでいる。押し以前の問題なので先にそちらを見る")
    print(f"[baseline] t={args.t_hit:.2f} s の骨盤 = ({pelvis_xy[0]:+.3f}, {pelvis_xy[1]:+.3f}, "
          f"{pelvis_z:.3f}) m  yaw={math.degrees(yaw):+.1f} deg  "
          f"押し後窓の最大傾き={baseline_tilt.max():.2f} deg")

    import mujoco  # noqa: F401  （robot_reach の中で使う。ここで存在確認だけしておく）

    results: list[Result] = []
    try:
        for direction in directions:
            body_dir = APPROACH_BODY[direction]
            # 体幹 yaw を掛けて世界座標へ。板はこの向きから来る。
            approach = np.array([
                body_dir[0] * math.cos(yaw) - body_dir[1] * math.sin(yaw),
                body_dir[0] * math.sin(yaw) + body_dir[1] * math.cos(yaw),
            ])
            reach = robot_reach(args.scene, baseline_q[hit_index], pelvis_xy, approach,
                                pelvis_z - half[2], pelvis_z + half[2])
            for impulse in impulses:
                speed = impulse / args.pusher_mass
                omega = math.pi / (2.0 * args.t_hit)
                travel = speed / omega
                stiffness = args.pusher_mass * omega * omega
                tag = f"{direction}_{impulse:g}"
                scene = scene_dir / f"{SCENE_PREFIX}{tag}.xml"
                qpos_csv = OUTPUT_DIR / f"push_{tag}.csv"
                # 板の中心 = 骨盤中心から板の側へ（張り出し + 板厚 + 助走 + 隙間）
                plate_xy = pelvis_xy + approach * (reach + half[0] + travel + 0.005)
                write_push_scene(scene, args.scene, tag,
                                 np.array([plate_xy[0], plate_xy[1], pelvis_z]),
                                 math.atan2(-approach[1], -approach[0]),
                                 -approach, travel, stiffness, args.pusher_mass, half)
                generated.append(scene)

                print(f"[run] {direction:5s} p={impulse:4.1f} N s  "
                      f"（板 {args.pusher_mass:.1f} kg を {speed:.2f} m/s、"
                      f"助走 {travel:.2f} m、k={stiffness:.3f} N/m）")
                evals = run_sim(args.sim, scene, qpos_csv, args.seconds,
                                args.command_x, args.command_yaw, args.qpos_hz, args.rl_start)
                condition = Condition(direction, impulse, speed, travel, stiffness,
                                      scene, qpos_csv)
                model = mujoco.MjModel.from_xml_path(str(scene))
                plate_adr = int(model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_slide")])
                result = analyze(condition, baseline_t, baseline_q, plate_adr,
                                 args.pusher_mass, args.t_hit, args.fall_height,
                                 args.fall_tilt, approach, evals)
                results.append(result)

                if not result.hit:
                    print("      板が当たっていない。--plate-half の幅を広げるか t-hit を見直す")
                if args.video:
                    VIDEO_DIR.mkdir(exist_ok=True)
                    render_video(qpos_csv, scene, VIDEO_DIR / f"push_{tag}.mp4",
                                 args.video_fps, args.qpos_hz)
                if result.fell and not args.continue_after_fall:
                    print(f"      転倒。{direction} はここで打ち切る")
                    break
    finally:
        if not args.keep_scene:
            for scene in generated:
                scene.unlink(missing_ok=True)

    # ------------------------------------------------------------------ 表
    print()
    print(f"指令 = (x {args.command_x:g}, yaw {args.command_yaw:g})   "
          f"押し時刻 = {args.t_hit:g} s   押し板 = {args.pusher_mass:g} kg   "
          f"転倒判定 = 骨盤高 < {args.fall_height:g} m または 傾き > {args.fall_tilt:g} deg")
    print(f"baseline（押し無し・同じ窓）の最大傾き = {baseline_tilt.max():.2f} deg / "
          f"最低骨盤高 = {baseline_q[baseline_after, 2].min():.4f} m / "
          f"tau_ratio_max = {baseline_evals.get('tau_ratio_max', '-')}")
    header = (f"{'方向':<6}{'公称p':>7}{'実測p':>8}{'Δv基部':>9}{'最大傾き':>10}"
              f"{'最低骨盤高':>11}{'τ最大':>8}{'変位':>8}  判定")
    print(header)
    print("-" * (len(header) + 8))
    for result in results:
        verdict = "転倒" if result.fell else ("復帰" if result.recovered else "耐えた(未収束)")
        if not result.hit:
            verdict = "空振り"
        print(f"{result.condition.direction:<6}{result.condition.impulse:>7.1f}"
              f"{result.delivered:>8.2f}{result.base_dv:>9.3f}{result.max_tilt_deg:>10.2f}"
              f"{result.min_pelvis_z:>11.4f}{result.tau_ratio_max:>8.3f}"
              f"{result.drift_m:>8.3f}  {verdict}")

    # 方向ごとの「耐えた最大の力積」。転倒しなければ最大値まで耐えたという意味。
    print()
    for direction in directions:
        survived = [r.condition.impulse for r in results
                    if r.condition.direction == direction and r.hit and not r.fell]
        fell = [r.condition.impulse for r in results
                if r.condition.direction == direction and r.fell]
        if fell:
            print(f"  {direction:<6} 耐えた最大 {max(survived) if survived else 0:.1f} N s / "
                  f"転倒 {min(fell):.1f} N s")
        else:
            print(f"  {direction:<6} 耐えた最大 {max(survived) if survived else 0:.1f} N s "
                  f"（試した範囲では転倒なし）")

    args.csv.parent.mkdir(exist_ok=True)
    with args.csv.open("w", encoding="utf-8") as handle:
        handle.write("direction,impulse_nominal_Ns,impulse_delivered_Ns,base_dv_mps,"
                     "max_tilt_deg,min_pelvis_z_m,tau_ratio_max,drift_m,hit,fell,recovered\n")
        for r in results:
            handle.write(f"{r.condition.direction},{r.condition.impulse:g},{r.delivered:.4f},"
                         f"{r.base_dv:.4f},{r.max_tilt_deg:.3f},{r.min_pelvis_z:.4f},"
                         f"{r.tau_ratio_max:.4f},{r.drift_m:.4f},"
                         f"{int(r.hit)},{int(r.fell)},{int(r.recovered)}\n")
    print(f"\n書いた: {args.csv}")

    return 1 if any(r.fell for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
