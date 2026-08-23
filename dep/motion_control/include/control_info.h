#pragma once

// 制御器の各モジュールが共有するストア。rabbit_mujoco_2 の control_info に対応する。
// モジュールは Invoke() の中でこのストアを読み書きするだけで、直接やり取りはしない。

#include <array>

#include "big_rabbit_model_param.h"

namespace big_rabbit::motion
{

    struct ControlInfo
    {
        // ---- 関節次元（policy と Isaac が見る座標系。MJCF の関節軸そのまま）----
        JointArray joint_position{};      // [rad]
        JointArray joint_velo{};          // [rad/s]
        JointArray joint_torque{};        // [N m] 実測
        JointArray joint_position_ref{};  // [rad] policy が出す目標
        JointArray joint_torque_ref{};    // [N m] ドライバ出力を関節側へ戻したもの

        // ---- モータ次元（ドライバが見る座標系。符号 x 減速比を掛けたもの）----
        MotorArray motor_angle{};         // [rad]
        MotorArray motor_velo{};          // [rad/s]
        MotorArray motor_angle_ref{};     // [rad] 62.5 Hz で更新され、1 kHz の間は保持される
        MotorArray motor_torque_ref{};    // [N m]

        // ---- 足裏の法線力（生値）----
        // 実機では足裏力センサ、sim では MuJoCo の接触力の集計。
        // 0/1 への閾値化は bridge が行う（学習側と同じ閾値を 1 箇所で持つため）。
        std::array<float, 2> foot_force_n{};

        // ---- 骨盤高 [m] ----
        // 対応するセンサが無い。sim では真値、実機では支持脚から運動学で逆算する。
        // observation の正規化 std が最小で、最も感度が高い項。
        float base_height_m = 0.0f;

        // ---- IMU ----
        std::array<float, 4> imu_orientation{1.0f, 0.0f, 0.0f, 0.0f};  // w, x, y, z
        std::array<float, 3> imu_angular_velocity{};
        std::array<float, 3> imu_linear_acceleration{};
    };

}  // namespace big_rabbit::motion
