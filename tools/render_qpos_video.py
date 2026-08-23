#!/usr/bin/env python3
"""big_rabbit_mujoco が吐いた qpos 軌跡を mp4 にする。

物理と制御は C++ 側が正本で、ここは描画だけを担う。
qpos を書き戻して mj_forward するだけなので、動画は必ず実行結果そのものになる。

    BIG_RABBIT_HEADLESS=1 BIG_RABBIT_MAX_SIM_TIME=11 BIG_RABBIT_MOTION_COMMAND_X=1 \
    BIG_RABBIT_QPOS_CSV=/tmp/qpos_forward.csv BIG_RABBIT_LOG_INTERVAL_S=0 \
      ./build-linux/big_rabbit_mujoco_sim
    python3 tools/render_qpos_video.py /tmp/qpos_forward.csv --out videos/forward.mp4

カメラは骨盤を追う。--fixed で固定カメラ。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "robotmodel" / "big_rabbit" / "scene.xml"


def find_ffmpeg() -> str:
    """system の ffmpeg が無い環境が多いので、imageio_ffmpeg の静的バイナリも探す。"""
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = list(
        Path("/home/pomiou/work/env_isaaclab/lib").glob(
            "python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
        )
    )
    if candidates:
        return str(candidates[0])
    raise SystemExit("ffmpeg が見つからない")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("qpos_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--distance", type=float, default=1.8)
    parser.add_argument("--azimuth", type=float, default=130.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--fixed", action="store_true", help="骨盤を追わず固定カメラにする")
    args = parser.parse_args()

    import mujoco

    with args.qpos_csv.open() as handle:
        reader = csv.reader(handle)
        next(reader)
        rows = np.array([[float(v) for v in row] for row in reader if row], dtype=np.float64)
    if rows.size == 0:
        raise SystemExit(f"qpos CSV が空: {args.qpos_csv}")
    time = rows[:, 0]
    qpos = rows[:, 1:]

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    if qpos.shape[1] != model.nq:
        raise SystemExit(f"nq 不一致: CSV {qpos.shape[1]} / model {model.nq}")

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    ffmpeg = find_ffmpeg()
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
        "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(args.out),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)

    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for index in range(len(time)):
            data.qpos[:] = qpos[index]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if not args.fixed:
                # 骨盤を追う。歩行では原点から離れていくので固定カメラだと画角から出る。
                pelvis = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
                camera.lookat[:] = [pelvis[0], pelvis[1], 0.30]
            renderer.update_scene(data, camera)
            encoder.stdin.write(renderer.render().astype(np.uint8).tobytes())

    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg が失敗した")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out}  ({len(time)} frames, {time[-1] - time[0]:.1f} s, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
