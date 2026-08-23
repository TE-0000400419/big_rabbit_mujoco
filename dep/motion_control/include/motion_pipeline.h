#pragma once

// パイプラインを構成する 4 モジュール。
//
//   UpdateModule            センサ（関節次元）-> ストア -> モータ次元へ変換
//   JointPositionTransfer   joint_position_ref -> motor_angle_ref（符号 x 減速比）
//   MotorDriverSim          motor_angle_ref -> motor_torque_ref（sim 専用。実機はファーム）
//   TorqueSet               motor_torque_ref -> joint_torque_ref（MuJoCo へ渡す形）
//
// 実行周期が 2 系統ある点に注意。
//   62.5 Hz: UpdateModule -> JointPositionTransfer   （policy と同じ周期。目標は次まで保持）
//    1 kHz : UpdateModule -> MotorDriverSim -> TorqueSet
// Isaac の実装 PD も target を 16 物理 step 保持して 1 kHz で効くので、これと一致する。

#include "motion_module.h"

namespace big_rabbit::motion
{

    /// センサ値をストアへ入れ、モータ次元へ変換する。
    class UpdateModule : public MotionModule
    {
    public:
        using MotionModule::MotionModule;
        int Invoke(const char *arg, int &param, NextState &result) noexcept override;
    };

    /// policy の関節位置目標をモータ位置目標へ落とす。
    class JointPositionTransfer : public MotionModule
    {
    public:
        using MotionModule::MotionModule;
        int Invoke(const char *arg, int &param, NextState &result) noexcept override;
    };

    /// モータドライバの位置制御ループ。**sim 専用**。実機ではドライバのファームに置き換わる。
    class MotorDriverSim : public MotionModule
    {
    public:
        using MotionModule::MotionModule;
        void Setup() noexcept override;
        int Invoke(const char *arg, int &param, NextState &result) noexcept override;
        int Reset(const char *arg, int &param, NextState &result) noexcept override;

        /// 直近 Invoke でのトルク上限使用率。飽和しているかの確認用。
        float torque_ratio() const noexcept { return torque_ratio_; }

    private:
        std::array<float, kJointNum> stiffness_{};
        std::array<float, kJointNum> damping_{};
        std::array<float, kJointNum> effort_limit_{};
        float torque_ratio_ = 0.0f;
    };

    /// ドライバ出力を関節側トルクへ戻す。MuJoCo の actuator へ渡す値になる。
    class TorqueSet : public MotionModule
    {
    public:
        using MotionModule::MotionModule;
        int Invoke(const char *arg, int &param, NextState &result) noexcept override;
    };

}  // namespace big_rabbit::motion
