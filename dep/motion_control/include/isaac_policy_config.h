#pragma once

// tools/export_isaac_policy_headers.py が生成。手で編集しないこと。
// Params: /home/pomiou/work/big_rabbit_isaac/configs/big_rabbit_balance/big_rabbit_walk_v24_robust_4096_500.params.yaml

#include <array>
#include <cstdlib>

namespace isaac_policy
{

    inline constexpr float kPi = 3.14159265358979323846f;

    // policy の入出力次元。isaac_policy_weights.h が static_assert で突き合わせる。
    inline constexpr int kObsDim = 54;
    inline constexpr int kActionDim = 10;
    inline constexpr int kHiddenDim = 128;
    inline constexpr float kObsNormalizerEps = 1.0e-2f;

    // 歩容位相の周波数。観測の gait_phase と参照テーブルの位相で共通。
    inline constexpr float kGaitFrequencyHz = 1.4286000000f;

    // policy 制御周期。物理 0.001 s x decimation 16。
    inline constexpr float kStepDt = 0.0160000000f;
    inline constexpr int kDecimation = 16;
    inline constexpr float kPhysicsDt = 0.0010000000f;

    // 接地観測の閾値。Isaac の ContactSensor と同じ意味。
    inline constexpr float kFootContactThresholdN = 1.000000f;

    // 起動時の骨盤高。
    inline constexpr float kSpawnHeightM = 0.430000f;

    inline constexpr std::array<const char *, 10> kJointNames = {
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_joint",
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_joint",
    };

    inline constexpr std::array<const char *, 2> kFootBodyNames = {
        "left_foot",
        "right_foot",
    };

    // 基準姿勢 crouch [rad]。hip+knee+ankle=0 で足裏が水平になる
    inline constexpr std::array<float, 10> kCrouchJointPositionRad = {
        +0.0000000000f,  // left_hip_yaw_joint
        -0.2094395102f,  // left_hip_roll_joint
        +1.2915436465f,  // left_hip_pitch_joint
        -1.6580627894f,  // left_knee_joint
        +0.3665191429f,  // left_ankle_joint
        +0.0000000000f,  // right_hip_yaw_joint
        +0.2094395102f,  // right_hip_roll_joint
        +1.2915436465f,  // right_hip_pitch_joint
        -1.6580627894f,  // right_knee_joint
        +0.3665191429f,  // right_ankle_joint
    };

    // action 1.0 あたりの関節角 [rad]
    inline constexpr std::array<float, 10> kJointActionScaleRad = {
        +0.1745329252f,  // left_hip_yaw_joint
        +0.1745329252f,  // left_hip_roll_joint
        +0.4363323130f,  // left_hip_pitch_joint
        +0.5235987756f,  // left_knee_joint
        +0.2617993878f,  // left_ankle_joint
        +0.1745329252f,  // right_hip_yaw_joint
        +0.1745329252f,  // right_hip_roll_joint
        +0.4363323130f,  // right_hip_pitch_joint
        +0.5235987756f,  // right_knee_joint
        +0.2617993878f,  // right_ankle_joint
    };

    // Isaac の実装 PD 剛性 [N m/rad]（関節側）
    inline constexpr std::array<float, 10> kJointStiffness = {
        +40.0000000000f,  // left_hip_yaw_joint
        +60.0000000000f,  // left_hip_roll_joint
        +60.0000000000f,  // left_hip_pitch_joint
        +45.0000000000f,  // left_knee_joint
        +30.0000000000f,  // left_ankle_joint
        +40.0000000000f,  // right_hip_yaw_joint
        +60.0000000000f,  // right_hip_roll_joint
        +60.0000000000f,  // right_hip_pitch_joint
        +45.0000000000f,  // right_knee_joint
        +30.0000000000f,  // right_ankle_joint
    };

    // Isaac の実装 PD 減衰 [N m s/rad]（関節側）
    inline constexpr std::array<float, 10> kJointDamping = {
        +2.0000000000f,  // left_hip_yaw_joint
        +3.5000000000f,  // left_hip_roll_joint
        +3.5000000000f,  // left_hip_pitch_joint
        +2.2000000000f,  // left_knee_joint
        +1.2000000000f,  // left_ankle_joint
        +2.0000000000f,  // right_hip_yaw_joint
        +3.5000000000f,  // right_hip_roll_joint
        +3.5000000000f,  // right_hip_pitch_joint
        +2.2000000000f,  // right_knee_joint
        +1.2000000000f,  // right_ankle_joint
    };

    // 関節側トルク上限 [N m]
    inline constexpr std::array<float, 10> kJointEffortLimitNm = {
        +23.7000000000f,  // left_hip_yaw_joint
        +33.5000000000f,  // left_hip_roll_joint
        +33.5000000000f,  // left_hip_pitch_joint
        +33.5000000000f,  // left_knee_joint
        +23.7000000000f,  // left_ankle_joint
        +23.7000000000f,  // right_hip_yaw_joint
        +33.5000000000f,  // right_hip_roll_joint
        +33.5000000000f,  // right_hip_pitch_joint
        +33.5000000000f,  // right_knee_joint
        +23.7000000000f,  // right_ankle_joint
    };

    // 関節側速度上限 [rad/s]
    inline constexpr std::array<float, 10> kJointVelocityLimitRadS = {
        +30.0000000000f,  // left_hip_yaw_joint
        +21.0000000000f,  // left_hip_roll_joint
        +21.0000000000f,  // left_hip_pitch_joint
        +21.0000000000f,  // left_knee_joint
        +30.0000000000f,  // left_ankle_joint
        +30.0000000000f,  // right_hip_yaw_joint
        +21.0000000000f,  // right_hip_roll_joint
        +21.0000000000f,  // right_hip_pitch_joint
        +21.0000000000f,  // right_knee_joint
        +30.0000000000f,  // right_ankle_joint
    };

    // 指令 (x, y, yaw)。環境変数で上書きできる。
    inline float EnvFloat(const char *name, float default_value)
    {
        const char *value = std::getenv(name);
        if (!value || value[0] == '\0')
        {
            return default_value;
        }
        char *end = nullptr;
        const float parsed = std::strtof(value, &end);
        return end == value ? default_value : parsed;
    }

    inline const std::array<float, 3> &MotionCommand()
    {
        static const std::array<float, 3> command = {
            EnvFloat("BIG_RABBIT_MOTION_COMMAND_X", 0.0f),
            EnvFloat("BIG_RABBIT_MOTION_COMMAND_Y", 0.0f),
            EnvFloat("BIG_RABBIT_MOTION_COMMAND_YAW", 0.0f),
        };
        return command;
    }

}  // namespace isaac_policy
