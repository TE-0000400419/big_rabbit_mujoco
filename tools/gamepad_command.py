#!/usr/bin/env python3
"""USB ゲームパッドを読んで、移動指令を UDP で制御プロセスへ送る。

制御プロセスとは別プロセスにしてある。理由は 2 つ。

  1. ゲームパッドのプロセスが落ちても制御プロセスは生き続け、
     受信側の watchdog が指令をゼロへ落とす（= 立ち止まる）ことを独立に保証できる。
  2. sim と実機で同じスクリプトを使える。sim では loopback、
     操作 PC を分ける構成でも送信先 IP を変えるだけ。

依存パッケージは無い。`/dev/input/js0` を 8 バイトずつ読むだけ。

    # 軸の割り当てを調べる
    python3 tools/gamepad_command.py --monitor

    # 送信（既定は 127.0.0.1:51234）
    python3 tools/gamepad_command.py

学習時の指令は 0 / ±1 の離散値だけだったので、**送信側で量子化する**。
そのため既定はボタン入力（4 ボタン）。スティックを使いたいときは --mode axes。

    既定のボタン割り当て
      1 前進   2 後退   0 左旋回   3 右旋回
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time
from pathlib import Path


# /dev/input/js0 の 1 イベントは 8 バイト: time(u32) value(i16) type(u8) number(u8)
JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
AXIS_FULL_SCALE = 32767.0


def checksum(payload: str) -> str:
    """NMEA 風の XOR チェックサム。UART へ載せ替えても同じ形で使える。"""
    value = 0
    for char in payload:
        value ^= ord(char)
    return f"{value:02X}"


def encode(seq: int, command: tuple[float, float, float]) -> bytes:
    payload = f"CMD,{seq % 65536},{command[0]:+.3f},{command[1]:+.3f},{command[2]:+.3f}"
    return f"{payload}*{checksum(payload)}\n".encode("ascii")


def quantize(value: float, dead_zone: float) -> float:
    """0 / ±1 へ落とす。学習時の指令が離散値だったので既定はこれ。"""
    if value > dead_zone:
        return 1.0
    if value < -dead_zone:
        return -1.0
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=os.environ.get("BIG_RABBIT_JOYSTICK", "/dev/input/js0"))
    parser.add_argument("--host", default=os.environ.get("BIG_RABBIT_COMMAND_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BIG_RABBIT_COMMAND_PORT", "51234")))
    parser.add_argument("--rate", type=float, default=50.0, help="送信レート [Hz]")
    parser.add_argument("--mode", choices=("buttons", "axes"), default="buttons",
                        help="buttons: 4 ボタンで離散指令（既定）。axes: スティック")
    parser.add_argument("--button-forward", type=int, default=1)
    parser.add_argument("--button-backward", type=int, default=2)
    parser.add_argument("--button-turnleft", type=int, default=0)
    parser.add_argument("--button-turnright", type=int, default=3)
    parser.add_argument("--axis-forward", type=int, default=1, help="前後に使う軸番号")
    parser.add_argument("--axis-lateral", type=int, default=0, help="左右に使う軸番号")
    parser.add_argument("--axis-yaw", type=int, default=3, help="旋回に使う軸番号")
    parser.add_argument("--invert-forward", type=int, default=1, help="1 なら符号反転（スティック上 = 前進）")
    parser.add_argument("--invert-yaw", type=int, default=1, help="1 なら符号反転（スティック左 = 左旋回）")
    parser.add_argument("--dead-zone", type=float, default=0.4, help="量子化のしきい値")
    parser.add_argument("--continuous", action="store_true",
                        help="量子化せずアナログ値を送る。学習分布外なので既定は off")
    parser.add_argument("--monitor", action="store_true", help="送信せず軸の値を表示する（割り当て調査用）")
    args = parser.parse_args()

    device = Path(args.device)
    if not device.exists():
        print(f"[FATAL] ゲームパッドが見つからない: {device}", file=sys.stderr)
        print("  接続を確認する。ls /dev/input/js*", file=sys.stderr)
        return 1

    axes: dict[int, float] = {}
    buttons: dict[int, int] = {}
    sock = None
    if not args.monitor:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[gamepad] {device} -> udp {args.host}:{args.port} @ {args.rate:.0f} Hz")
        if args.mode == "buttons":
            print(f"[gamepad] ボタン {args.button_forward}=前進 {args.button_backward}=後退 "
                  f"{args.button_turnleft}=左旋回 {args.button_turnright}=右旋回")
        else:
            print(f"[gamepad] 前後=軸{args.axis_forward} 旋回=軸{args.axis_yaw} "
                  f"量子化={'off（アナログ）' if args.continuous else f'on（しきい値 {args.dead_zone}）'}")
    else:
        print(f"[gamepad] monitor: {device}（Ctrl-C で終了）")

    seq = 0
    interval = 1.0 / max(args.rate, 1.0)
    next_send = time.monotonic()

    # ノンブロッキングで開き、溜まったイベントを読み切ってから送信する。
    with open(device, "rb", buffering=0) as handle:
        os.set_blocking(handle.fileno(), False)
        try:
            while True:
                while True:
                    chunk = handle.read(JS_EVENT_SIZE)
                    if not chunk or len(chunk) < JS_EVENT_SIZE:
                        break
                    _, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, chunk)
                    # 初期化イベント（0x80）は現在値の通知なので同じ扱いでよい。
                    kind = event_type & ~JS_EVENT_INIT
                    if kind == JS_EVENT_AXIS:
                        axes[number] = value / AXIS_FULL_SCALE
                    elif kind == JS_EVENT_BUTTON:
                        buttons[number] = value

                now = time.monotonic()
                if now < next_send:
                    time.sleep(min(0.002, next_send - now))
                    continue
                next_send += interval

                if args.monitor:
                    axis_text = " ".join(f"{i}:{v:+.2f}" for i, v in sorted(axes.items()))
                    pressed = [i for i, v in sorted(buttons.items()) if v]
                    print(f"\rボタン {pressed}   軸 {axis_text}          ", end="", flush=True)
                    continue

                if args.mode == "buttons":
                    # 4 ボタンの離散指令。学習時の指令（0 / ±1）と完全に一致する。
                    # 相反するボタンが同時に押されたら 0 にする（暴走より停止を選ぶ）。
                    forward = float(bool(buttons.get(args.button_forward))) \
                        - float(bool(buttons.get(args.button_backward)))
                    yaw = float(bool(buttons.get(args.button_turnleft))) \
                        - float(bool(buttons.get(args.button_turnright)))
                    command = (forward, 0.0, yaw)
                else:
                    raw_forward = axes.get(args.axis_forward, 0.0) * (-1.0 if args.invert_forward else 1.0)
                    raw_lateral = axes.get(args.axis_lateral, 0.0)
                    raw_yaw = axes.get(args.axis_yaw, 0.0) * (-1.0 if args.invert_yaw else 1.0)
                    if args.continuous:
                        command = (
                            max(-1.0, min(1.0, raw_forward)),
                            max(-1.0, min(1.0, raw_lateral)),
                            max(-1.0, min(1.0, raw_yaw)),
                        )
                    else:
                        command = (
                            quantize(raw_forward, args.dead_zone),
                            quantize(raw_lateral, args.dead_zone),
                            quantize(raw_yaw, args.dead_zone),
                        )

                sock.sendto(encode(seq, command), (args.host, args.port))
                seq += 1
        except KeyboardInterrupt:
            print()
            if sock is not None:
                # 終了時にゼロ指令を数回送って確実に止める。
                # 送れなくても受信側の watchdog が止めるが、両方あるほうがよい。
                for _ in range(5):
                    sock.sendto(encode(seq, (0.0, 0.0, 0.0)), (args.host, args.port))
                    seq += 1
                    time.sleep(0.01)
                print("[gamepad] ゼロ指令を送って終了した")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
