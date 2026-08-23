#pragma once

// 関節次元 <-> モータ次元の変換。
//
//   theta_motor = s * N * theta_joint      omega_motor = s * N * omega_joint
//   tau_joint   = s * N * tau_motor        （減速機がトルクを N 倍する）
//
// s は 00_model.txt の関節駆動方向補正、N は減速比。
// 位置とトルクの両方に一貫して s を掛けることが必須。片方だけだと正フィードバックになる。

#include <array>
#include <cstddef>

#include "big_rabbit_model_param.h"

namespace big_rabbit
{

    inline void Joint2MotorPosition(const JointArray &joint_rad, MotorArray &motor_rad) noexcept
    {
        for (std::size_t i = 0; i < kJointNum; i++)
        {
            motor_rad[i] = kJointDirection[i] * kGearRatio[i] * joint_rad[i];
        }
    }

    inline void Joint2MotorVelocity(const JointArray &joint_rad_s, MotorArray &motor_rad_s) noexcept
    {
        for (std::size_t i = 0; i < kJointNum; i++)
        {
            motor_rad_s[i] = kJointDirection[i] * kGearRatio[i] * joint_rad_s[i];
        }
    }

    inline void Motor2JointPosition(const MotorArray &motor_rad, JointArray &joint_rad) noexcept
    {
        for (std::size_t i = 0; i < kJointNum; i++)
        {
            joint_rad[i] = motor_rad[i] / (kJointDirection[i] * kGearRatio[i]);
        }
    }

    inline void Motor2JointVelocity(const MotorArray &motor_rad_s, JointArray &joint_rad_s) noexcept
    {
        for (std::size_t i = 0; i < kJointNum; i++)
        {
            joint_rad_s[i] = motor_rad_s[i] / (kJointDirection[i] * kGearRatio[i]);
        }
    }

    inline void Motor2JointTorque(const MotorArray &motor_torque_nm, JointArray &joint_torque_nm) noexcept
    {
        for (std::size_t i = 0; i < kJointNum; i++)
        {
            joint_torque_nm[i] = kJointDirection[i] * kGearRatio[i] * motor_torque_nm[i];
        }
    }

    inline void Joint2MotorTorque(const JointArray &joint_torque_nm, MotorArray &motor_torque_nm) noexcept
    {
        for (std::size_t i = 0; i < kJointNum; i++)
        {
            motor_torque_nm[i] = joint_torque_nm[i] / (kJointDirection[i] * kGearRatio[i]);
        }
    }

}  // namespace big_rabbit
