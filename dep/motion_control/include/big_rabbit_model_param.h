#pragma once

// Big Rabbit のメカ諸元。出典は big_rabbit_isaac/big_rabbit_model/robot_model/00_model.txt。
// policy 側の数値（crouch / action scale / PD ゲイン）は isaac_policy_config.h（生成物）にある。
// ここには「Isaac には出てこないがドライバに必要な情報」だけを置く。

#include <array>
#include <cstddef>

#include "isaac_policy_config.h"

namespace big_rabbit
{

    inline constexpr std::size_t kJointNum = 10;

    // 関節次元 / モータ次元の配列。要素順は isaac_policy::kJointNames と同じ。
    using JointArray = std::array<float, kJointNum>;
    using MotorArray = std::array<float, kJointNum>;

    // ---- 減速比 ----
    // J1 hip_yaw / J5 ankle は GO M8010-6（6.33:1）、J2-J4 は Unitree A1（9.1:1）。
    inline constexpr float kGearGo = 6.33f;
    inline constexpr float kGearA1 = 9.10f;

    inline constexpr std::array<float, kJointNum> kGearRatio = {
        kGearGo,  // left_hip_yaw
        kGearA1,  // left_hip_roll
        kGearA1,  // left_hip_pitch
        kGearA1,  // left_knee
        kGearGo,  // left_ankle
        kGearGo,  // right_hip_yaw
        kGearA1,  // right_hip_roll
        kGearA1,  // right_hip_pitch
        kGearA1,  // right_knee
        kGearGo,  // right_ankle
    };

    // ---- 関節駆動方向補正 ----
    // ロボットモデル定義の座標と world の回転方向が合わないぶんの符号（00_model.txt）。
    // J3/J4/J5 は左右で符号が逆。MJCF の関節軸は左右同符号（world 基準）に揃えてあるので、
    // この符号差は純粋にモータ結線側の話。取り違えると脚が逆に動く。
    inline constexpr std::array<float, kJointNum> kJointDirection = {
        -1.0f,  // left_hip_yaw    J1
        +1.0f,  // left_hip_roll   J2
        -1.0f,  // left_hip_pitch  J3
        -1.0f,  // left_knee       J4
        +1.0f,  // left_ankle      J5
        -1.0f,  // right_hip_yaw   J1
        +1.0f,  // right_hip_roll  J2
        +1.0f,  // right_hip_pitch J3
        +1.0f,  // right_knee      J4
        -1.0f,  // right_ankle     J5
    };

    // ---- モータドライバのゲイン ----
    // ドライバの制御則は tau_motor = Kp * dq_motor + Kd * dq_motor_dot。
    // Isaac で成立しているゲインは関節側なので、モータ次元へ落とす。
    //
    //   theta_motor = s * N * theta_joint,  tau_joint = s * N * tau_motor
    //   => tau_joint = s*N * Kp_m * (s*N*dq_joint) = N^2 * Kp_m * dq_joint
    //   => Kp_m = Kp_joint / N^2
    //
    // s^2 = 1 なので、符号を位置とトルクの両方に一貫して掛ければ関節等価ゲインは符号に依存しない。
    // 片方だけ掛けると正フィードバックになって発散する。
    inline constexpr float MotorStiffness(std::size_t index)
    {
        return isaac_policy::kJointStiffness[index] / (kGearRatio[index] * kGearRatio[index]);
    }

    inline constexpr float MotorDamping(std::size_t index)
    {
        return isaac_policy::kJointDamping[index] / (kGearRatio[index] * kGearRatio[index]);
    }

    // ---- モータ側の上限 ----
    // 00_model.txt の「最大出力トルク / 最大出力速度」は減速後（関節側）の値。
    // モータ側に直すと GO/A1 ともおよそ 3.7 N m / 190 rad/s に揃う。
    inline constexpr float MotorEffortLimitNm(std::size_t index)
    {
        return isaac_policy::kJointEffortLimitNm[index] / kGearRatio[index];
    }

    inline constexpr float MotorVelocityLimitRadS(std::size_t index)
    {
        return isaac_policy::kJointVelocityLimitRadS[index] * kGearRatio[index];
    }

    // ---- 関節側等価慣性 ----
    // 00_model.txt では GO / A1 ともアーマチュア慣性 0.01 kg m^2（関節側等価、Unitree 公式値）。
    // MJCF の armature にも同じ値が入っている。
    inline constexpr float kJointArmature = 0.01f;

}  // namespace big_rabbit
