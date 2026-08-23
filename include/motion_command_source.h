#pragma once

// 移動指令 (x, y, yaw) の入力元。
//
// 指令は「操作者の意図」なので sim と実機で同じ役割を持つが、**入力デバイスは違う**。
// そのため bridge には入れず、sim 側（実機では操作系）に置いて差し替え可能にする。
//
//   env      環境変数で固定。既定。再現性のある評価に使う
//   gamepad  USB ゲームパッド（GLFW の gamepad API）。手動操作
//   sequence 時刻で切り替わる指令列。指令切替の再現テストに使う
//
// bridge 側は `SetMotionCommand()` を毎 policy step 受け付けるので、どの入力元でも
// 実行中に変えてよい。指令切替への耐性は既に実証済み（24 s / 8 回の切替で転倒 0）。

#include <array>
#include <memory>
#include <string>
#include <vector>

class MotionCommandSource
{
public:
    virtual ~MotionCommandSource() = default;

    /// 毎 policy step 呼ぶ。time_s は制御開始からの経過時刻。
    virtual void Poll(double time_s) = 0;
    virtual std::array<float, 3> command() const = 0;
    virtual const char *name() const = 0;
    /// 起動時に 1 行だけ状態を出す（接続の有無など）。
    virtual std::string status() const { return {}; }
};

/// 環境変数で入力元を選ぶ。
///   BIG_RABBIT_COMMAND_SOURCE = env | gamepad | sequence
///   BIG_RABBIT_COMMAND_SEQUENCE = "forward:4,stop:2,backward:4,..."（sequence のとき）
///   BIG_RABBIT_COMMAND_CONTINUOUS = 1 で gamepad をアナログ値のまま渡す
std::unique_ptr<MotionCommandSource> MakeMotionCommandSourceFromEnv();
