#include "motion_pipeline.h"

#include <algorithm>
#include <cmath>

#include "joint_motor_transform.h"

namespace big_rabbit::motion
{

    int UpdateModule::Invoke(const char *arg, int &param, NextState &result) noexcept
    {
        (void)arg;
        (void)param;
        // 関節次元のセンサ値は bridge が既にストアへ入れている。
        // ここではドライバが使うモータ次元へ変換するだけ。
        Joint2MotorPosition(control_info_->joint_position, control_info_->motor_angle);
        Joint2MotorVelocity(control_info_->joint_velo, control_info_->motor_velo);
        result = NextState::OK;
        return 0;
    }

    int JointPositionTransfer::Invoke(const char *arg, int &param, NextState &result) noexcept
    {
        (void)arg;
        (void)param;
        // policy が出した関節位置目標をモータ位置目標へ落とす。
        // これは 62.5 Hz でしか更新されず、次の更新まで 1 kHz のドライバが同じ値を追う。
        Joint2MotorPosition(control_info_->joint_position_ref, control_info_->motor_angle_ref);
        result = NextState::OK;
        return 0;
    }

    void MotorDriverSim::Setup() noexcept
    {
        // ゲインは Isaac で成立している関節側の値を N^2 で割ってモータ次元へ落としたもの。
        // 実機ではドライバに同じ値を設定する。
        for (std::size_t i = 0; i < kJointNum; i++)
        {
            stiffness_[i] = MotorStiffness(i);
            damping_[i] = MotorDamping(i);
            effort_limit_[i] = MotorEffortLimitNm(i);
        }
    }

    int MotorDriverSim::Invoke(const char *arg, int &param, NextState &result) noexcept
    {
        (void)arg;
        (void)param;
        // ドライバの位置制御ループ: tau = Kp * dq + Kd * dq_dot。
        // 入出力はモータ次元。1 kHz で回す。
        float worst_ratio = 0.0f;
        for (std::size_t i = 0; i < kJointNum; i++)
        {
            const float error = control_info_->motor_angle_ref[i] - control_info_->motor_angle[i];
            float torque = stiffness_[i] * error - damping_[i] * control_info_->motor_velo[i];
            const float limit = effort_limit_[i];
            torque = std::clamp(torque, -limit, limit);
            control_info_->motor_torque_ref[i] = torque;
            worst_ratio = std::max(worst_ratio, std::abs(torque) / limit);
        }
        torque_ratio_ = worst_ratio;
        result = NextState::OK;
        return 0;
    }

    int MotorDriverSim::Reset(const char *arg, int &param, NextState &result) noexcept
    {
        (void)arg;
        (void)param;
        control_info_->motor_torque_ref.fill(0.0f);
        torque_ratio_ = 0.0f;
        result = NextState::OK;
        return 0;
    }

    int TorqueSet::Invoke(const char *arg, int &param, NextState &result) noexcept
    {
        (void)arg;
        (void)param;
        Motor2JointTorque(control_info_->motor_torque_ref, control_info_->joint_torque_ref);
        result = NextState::OK;
        return 0;
    }

}  // namespace big_rabbit::motion
