#pragma once

// 制御モジュールの共通インタフェース。rabbit_mujoco_2 の MotionControl 系に対応する。
//
// Big Rabbit は位置制御ループをドライバが持つので、Rabbit にあった
// 「関節位置 -> 関節加速度 -> モータトルク」の 2 段（MotorPosControl / MotorAccControl）は無い。
// ただし制御系からの見た目を統一するため、Invoke で駆動するパイプライン方式は維持する。
//
// 実機との差異は MotorDriverSim だけ。実機ではドライバのファームがその役を担う。

#include "control_info.h"

namespace big_rabbit::motion
{

    enum class NextState
    {
        OK,
        Error,
    };

    class MotionModule
    {
    public:
        explicit MotionModule(ControlInfo *control_info) noexcept : control_info_(control_info) {}
        virtual ~MotionModule() = default;

        virtual void Setup() noexcept {}
        virtual int Invoke(const char *arg, int &param, NextState &result) noexcept = 0;
        virtual int Reset(const char *arg, int &param, NextState &result) noexcept
        {
            (void)arg;
            (void)param;
            result = NextState::OK;
            return 0;
        }

    protected:
        ControlInfo *control_info_ = nullptr;
    };

}  // namespace big_rabbit::motion
