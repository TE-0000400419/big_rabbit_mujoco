# big_rabbit_mujoco

`big_rabbit_isaac` で学習した policy を、**実機と同じ制御スタック越しに** MuJoCo で検証する。
`rabbit_isaac` → `rabbit_mujoco_2` の関係に対応する sim2sim 環境。

設計の詳細は
[big_rabbit_isaac/docs/big_rabbit_mujoco_design_ja.md](https://github.com/TE-0000400419/big_rabbit_isaac/blob/main/docs/big_rabbit_mujoco_design_ja.md)。

## rabbit_mujoco_2 との違い

Big Rabbit は**位置制御ループをモータドライバが持つ**ので、Rabbit にあった
加速度制御と外乱オブザーバの 2 層が要らない。

```
Rabbit:
  joint_ref -> MotorPosControl -> joint_accel_ref -> MotorAccControl(+DOB)
            -> MotionTransfer(Motor,Torque) -> TorqueSet -> MuJoCo ctrl

Big Rabbit:
  joint_ref -> JointPositionTransfer(符号 x 減速比) -> motor_angle_ref
            -> [sim] MotorDriverSim(Kp,Kd) -> motor_torque_ref
            -> TorqueSet(Motor2JointTorque) -> MuJoCo ctrl
```

その結果 SDRDtk / Controllers / PosDOB への依存が無く、**MuJoCo と GLFW だけでビルドできる**。
`Invoke` で駆動するパイプライン方式は Rabbit と揃えてある。

`MotorDriverSim` が実機と唯一違う箇所。実機ではドライバのファームに置き換わるので、
**単独で差し替えられる 1 モジュール**にしてある。

## sim と実機の境界

**bridge に入るのはすべてセンサ生値。** 差し替えるのは「生値をどこから取るか」だけ。

```
入力（すべて生値）
  SetSensorDataMotor(angle, velo, torque)   実機（USB）。内部で符号 x 減速比を戻す
  SetSensorDataJoint(pos, velo, torque)     sim（MuJoCo の qpos/qvel）
  SetSensorDataIMU(quat, gyro, accel)       両方。bridge が quaternion から重力方向を作る
  SetSensorDataFootForce(left_N, right_N)   両方。bridge が 0/1 へ閾値化
  SetSensorDataBaseHeight(m)                当面 sim 真値。将来 bridge 内の運動学
  SetMotionCommand(x, y, yaw)               実行中に変えてよい

出力
  GetMotorAngleRef(motor_angle_ref)         **実機の出力**。USB コントローラへ
  GetControlCommandsJoint(joint_torque)     sim 専用。MuJoCo の actuator へ
```

生値 -> observation の変換（`sensor_transform.h`）は **sim と実機で同一**なので bridge が持つ。
実機で再実装せずに済み、2 実装がずれる余地も無い。

**実機の出力はモータトルクではなくモータ位置目標。** ドライバが位置ループを持つ。
`ExecuteDriverStep()`（モータ PD）は sim 専用で、実機ではドライバのファームが担う。

**重力方向は加速度センサからは取らない。** IMU の姿勢 quaternion から計算する。
加速度センサ単体では機体自身の加速度と重力が区別できず、歩行中は使えない。

実機化の作業は原理的に `src/sim_main.cc` を USB HAL に差し替えるだけになる。

## policy が並進速度を見ていない

`base_lin_vel` は observation に入っていない。**実機に速度推定器が要らない。**

速度は報酬でだけ使い、推論では測らない。option B が速度生成をフィードフォワードへ移した
（参照テーブルが歩幅 0.14 m / 周期 0.7 s = 0.2 m/s を作る）ため、policy は参照からの逸脱を
補正するだけでよくなった。実測 0.203 m/s が公称 0.200 m/s と一致するのはこのため。

限界は**閉ループの速度調節ができない**こと。坂道や積載で速度がずれても補正できない。
必要になったら参照の振幅（`amp`）を外側から連続値で与える速度ループを足せる（**再学習不要**）。

詳細は設計書の 2.2.1 節。

## 周期

| 対象 | 周期 |
|---|---|
| 物理 | 1 kHz |
| モータドライバ | 1 kHz |
| policy | 62.5 Hz（decimation 16）|

Isaac の実装 PD も target を 16 物理 step 保持して 1 kHz で効くので、これで一致する。

## 使い方

```bash
make            # ビルド
make export     # Isaac の checkpoint からヘッダを生成（policy を差し替えるとき）
make check      # 単体テスト -> クロス検証 -> 保持 -> 追従 -> 歩行
make video      # 5 条件の mp4
make gui        # GUI 実行（DISPLAY が必要）
```

policy を差し替えるときは Makefile 冒頭の `POLICY_RUN` と `POLICY_PARAMS` を変えて `make export`。

### 環境変数

| 変数 | 既定 | 内容 |
|---|---|---|
| `BIG_RABBIT_SCENE_XML` | `robotmodel/big_rabbit/scene.xml` | シーン |
| `BIG_RABBIT_CONTROL_MODE` | `rl` | `rl` / `home`（crouch 保持）/ `reference`（参照そのまま）|
| `BIG_RABBIT_MOTION_COMMAND_X` | 0 | 前後指令。1 で前進、-1 で後退 |
| `BIG_RABBIT_MOTION_COMMAND_Y` | 0 | 横指令（学習では常に 0）|
| `BIG_RABBIT_MOTION_COMMAND_YAW` | 0 | 旋回指令。±1 |
| `BIG_RABBIT_MAX_SIM_TIME` | 0（無制限）| 実行秒数 |
| `BIG_RABBIT_RL_START_S` | 1.0 | RL に切り替える時刻。それまでは crouch |
| `BIG_RABBIT_HEADLESS` | 未設定 | 設定すると描画せず実時間同期もしない |
| `BIG_RABBIT_LOG_INTERVAL_S` | 1.0 | 進捗ログ間隔。0 で無効 |
| `BIG_RABBIT_TRACE_CSV` | — | 指令と実測を CSV に吐く（追従解析用）|
| `BIG_RABBIT_QPOS_CSV` | — | qpos 軌跡を CSV に吐く（動画用）|
| `BIG_RABBIT_QPOS_HZ` | 50 | qpos の記録レート |
| `BIG_RABBIT_REALTIME` | 未設定 | headless でも実時間に同期する（ゲームパッド操作用）|
| `BIG_RABBIT_COMMAND_SOURCE` | `env` | `env` / `udp` / `sequence` |
| `BIG_RABBIT_COMMAND_PORT` | 51234 | UDP 受信ポート |
| `BIG_RABBIT_COMMAND_TIMEOUT_S` | 0.2 | watchdog |
| `BIG_RABBIT_COMMAND_QUANTIZE` | 1 | 受信側でも 0/±1 に落とす |
| `BIG_RABBIT_COMMAND_SEQUENCE` | — | `sequence` の指令列 |

## 移動指令の入力

指令源は 3 つあり、`BIG_RABBIT_COMMAND_SOURCE` で選ぶ。

| 値 | 内容 |
|---|---|
| `env`（既定）| 環境変数で固定。再現性のある評価に使う |
| `udp` | **ゲームパッド**。別プロセスが UDP で送ってくる |
| `sequence` | 時刻で切り替わる指令列。指令切替の再現テスト |

### ゲームパッド

読み取りは**別プロセス**（`tools/gamepad_command.py`）にしてある。理由は 2 つ。

1. 送信側が落ちても制御プロセスは生き続け、**受信側の watchdog が指令をゼロへ落とす**
   （= 立ち止まる）ことを独立に保証できる
2. sim と実機で同じスクリプトが使える。sim では loopback、操作 PC を分けるなら送信先 IP を変えるだけ

依存パッケージは無い（`/dev/input/js0` を直読み）。

```bash
# 1. 軸・ボタンの割り当てを調べる（コントローラごとに違う）
python3 tools/gamepad_command.py --monitor

# 2. 送信（別端末）
python3 tools/gamepad_command.py

# 3. 制御側（GUI）
BIG_RABBIT_COMMAND_SOURCE=udp make gui

# 3'. 制御側（描画なし・実時間）
BIG_RABBIT_HEADLESS=1 BIG_RABBIT_REALTIME=1 BIG_RABBIT_COMMAND_SOURCE=udp \
  ./build-linux/big_rabbit_mujoco_sim
```

既定のボタン割り当て:

| ボタン | 指令 |
|---:|---|
| 1 | 前進 |
| 2 | 後退 |
| 0 | 左旋回 |
| 3 | 右旋回 |

4 ボタンの離散入力なので**学習時の指令（0 / ±1）と完全に一致する**。
相反するボタンが同時に押されたら 0（停止）にする。
スティックを使うなら `--mode axes`、アナログ通しは `--continuous`（学習分布外）。

### プロトコル

NMEA 風の 1 行 ASCII + XOR チェックサム。`echo` で手打ちでき、UART へ載せ替えても同じ形。

```
CMD,<seq>,<x>,<y>,<yaw>*<XX>\n
例: CMD,142,+1.000,+0.000,-0.350*3F
```

USB と UDP は別のレイヤ。ゲームパッドは USB でカーネルに繋がり `/dev/input/js0` として見える。
UDP は**プロセス間の通信手段**で、同一マシンなら loopback を通るだけ（ネットワーク機器は関与しない）。

### 安全設計

| 項目 | 挙動 |
|---|---|
| watchdog | 200 ms 有効フレームなし → 指令 (0,0,0)。`BIG_RABBIT_COMMAND_TIMEOUT_S` |
| チェックサム不一致 | そのフレームを捨てる |
| 最新優先 | 溜まった分は読み切って最後のフレームだけ採用 |
| 起動時 | 指令 (0,0,0)。フレームが来るまで crouch |
| 送信側の終了 | ゼロ指令を 5 回撃ってから終了 |

**指令ゼロは参照振幅 0 = crouch へ退化する**ので、「通信が切れたら立ち止まる」が構造的に保証される。
option B の副産物だが実機安全上かなり有利な性質。

## 検証結果（v21、2026-08-17）

クロス検証は 6 段。**学習側の数値と一致することを段ごとに数値で確認する。**
これを飛ばすと、動かない原因が観測の並びなのか制御器なのか切り分けられない。

| 段 | 内容 | 結果 |
|---|---|---|
| 1 | obs54 の一致 | 最大差 **3.58e-07**（許容 1e-5）|
| 2 | action10 の一致 | **1.43e-06**（許容 1e-4）|
| 3 | 参照テーブル補間 | **2.40e-07**（許容 1e-6）、位相 **0.00e+00** |
| 4 | 静止保持 | 骨盤高 0.3864 m / 直立度 0.9953 / トルク 7.8%（MuJoCo 素の PD と一致）|
| 5 | 追従性能 | hip 0.814（基準 0.809）/ knee 0.858（0.857）、遅れ差 1.1 deg 以内 |
| 6 | 歩行 | 下表 |

### 段 6: 歩行（10 s、5 条件とも転倒 0）

| 指標 | Isaac (v21) | big_rabbit_mujoco |
|---|---:|---:|
| 前進速度 | 0.205 m/s | **0.203** |
| 後退速度 | -0.182 | **-0.170** |
| 左旋回 | 0.601 rad/s | **0.578** |
| 右旋回 | -0.562 | **-0.652** |
| 接地率（前進）| 0.539 | **0.543** |
| 静止時 接地率 | 0.998 | **1.000** |
| 静止時 速度 | 0.000 | **-0.002** |

## 実装で注意した点

**歩容位相は自由走行クロック。接地でリセットしない。**
Big Rabbit は関節目標そのものが位相の関数（`joint_ref = q_ref(phi) + scale * action`）なので、
位相を飛ばすと参照が不連続になる。学習側で 0.03 m の不連続が転倒の主因になった実測がある
（直すだけで転倒率 1.25% -> 0.32%）。

**符号と減速比は位置とトルクの両方に一貫して掛ける。**
`s^2 = 1` で打ち消えるので関節等価ゲインは符号に依存しないが、片方だけ掛けると
正フィードバックになって発散する。`make test` の往復変換テストで潰す。

**`data.ctrl` はモータ側トルク。** MJCF の actuator は `gear` に減速比が入っているので
`tau_joint = gear * ctrl`。関節側トルクをそのまま書くと静かに `gear` 倍（6.33/9.1）になる。

**observation は `obs[n]` へ明示代入する。** `obs_index++` にすると並びのずれが見えなくなる。

## 未着手

| 項目 | 内容 |
|---|---|
| `base_height` の運動学 | 現在は MuJoCo の pelvis world z。実機では支持脚から逆算する |
| 右旋回が 116% | -0.652 vs Isaac -0.562。単一試行なので複数シードで確認したい |
| トルク使用率 0.71（旋回）| 静止 7.7% に対して高い。原因未特定 |
| action latency | 学習側で option B の分岐により無効。Phase 6 の課題 |
| ドライバの実行周期・フィルタ | 1 kHz と仮定。実機ファームの仕様待ち |
