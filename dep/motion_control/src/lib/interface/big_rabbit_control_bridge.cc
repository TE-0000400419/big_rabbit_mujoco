#include "big_rabbit_control_bridge.h"

#include <cmath>

#include "control_info.h"
#include "isaac_policy_config.h"
#include "isaac_policy_infer.h"
#include "joint_motor_transform.h"
#include "motion_pipeline.h"
#include "phase_reference.h"
#include "sensor_transform.h"

using big_rabbit::motion::ControlInfo;
using big_rabbit::motion::JointPositionTransfer;
using big_rabbit::motion::MotorDriverSim;
using big_rabbit::motion::NextState;
using big_rabbit::motion::TorqueSet;
using big_rabbit::motion::UpdateModule;

namespace
{
    ControlInfo g_control_info;

    UpdateModule *g_update_module = nullptr;
    JointPositionTransfer *g_joint_position_transfer = nullptr;
    MotorDriverSim *g_motor_driver = nullptr;
    TorqueSet *g_torque_set = nullptr;

    BigRabbitControlBridge::IsaacPolicyDebugData g_debug_data;
    std::array<float, 3> g_motion_command{};

    // RSL-RL の actor 観測には action 履歴が 2 段入っている。
    // previous_action が 2 step 前、current_action が 1 step 前。初回はどちらもゼロ。
    std::array<float, BigRabbitControlBridge::kJointAxisNum> g_previous_action{};
    std::array<float, BigRabbitControlBridge::kJointAxisNum> g_current_action{};
} // namespace

namespace
{
    /// センサ生値から重力方向を作る。IMU の姿勢 quaternion から。
    std::array<float, 3> ProjectedGravityFromStore() noexcept
    {
        return big_rabbit::sensor::ProjectedGravityBody(g_control_info.imu_orientation);
    }

    /// センサ生値から接地 0/1 を作る。学習側と同じ閾値。
    std::array<float, 2> FeetContactFromStore() noexcept
    {
        return {
            big_rabbit::sensor::FootContactFromForce(g_control_info.foot_force_n[0],
                                                     isaac_policy::kFootContactThresholdN),
            big_rabbit::sensor::FootContactFromForce(g_control_info.foot_force_n[1],
                                                     isaac_policy::kFootContactThresholdN),
        };
    }
} // namespace

void BigRabbitControlBridge::Initialize() noexcept
{
    g_control_info = ControlInfo{};
    g_motion_command = isaac_policy::MotionCommand();
    // 起動時は crouch を目標にしておく。
    for (std::size_t i = 0; i < kJointAxisNum; i++)
    {
        g_control_info.joint_position_ref[i] = isaac_policy::kCrouchJointPositionRad[i];
    }
}

void BigRabbitControlBridge::ExecuteReady() noexcept
{
    g_update_module = new UpdateModule(&g_control_info);
    g_joint_position_transfer = new JointPositionTransfer(&g_control_info);
    g_motor_driver = new MotorDriverSim(&g_control_info);
    g_torque_set = new TorqueSet(&g_control_info);

    g_update_module->Setup();
    g_joint_position_transfer->Setup();
    g_motor_driver->Setup();
    g_torque_set->Setup();

    int param = 0;
    NextState result = NextState::OK;
    g_motor_driver->Reset("", param, result);

    g_previous_action.fill(0.0f);
    g_current_action.fill(0.0f);
}

void BigRabbitControlBridge::SetSensorDataJoint(
    const std::array<float, kJointAxisNum> &joint_positions_rad,
    const std::array<float, kJointAxisNum> &joint_velocities_rad_s,
    const std::array<float, kJointAxisNum> &joint_torques_Nm) noexcept
{
    for (std::size_t i = 0; i < kJointAxisNum; i++)
    {
        g_control_info.joint_position[i] = joint_positions_rad[i];
        g_control_info.joint_velo[i] = joint_velocities_rad_s[i];
        g_control_info.joint_torque[i] = joint_torques_Nm[i];
    }
}

void BigRabbitControlBridge::SetSensorDataMotor(
    const std::array<float, kJointAxisNum> &motor_positions_rad,
    const std::array<float, kJointAxisNum> &motor_velocities_rad_s,
    const std::array<float, kJointAxisNum> &motor_torques_Nm) noexcept
{
    // 実機（USB コントローラ）はモータ次元で来る。符号 x 減速比を戻して関節次元へ展開する。
    g_control_info.motor_angle = motor_positions_rad;
    g_control_info.motor_velo = motor_velocities_rad_s;
    big_rabbit::Motor2JointPosition(motor_positions_rad, g_control_info.joint_position);
    big_rabbit::Motor2JointVelocity(motor_velocities_rad_s, g_control_info.joint_velo);
    big_rabbit::Motor2JointTorque(motor_torques_Nm, g_control_info.joint_torque);
}

void BigRabbitControlBridge::SetSensorDataIMU(
    const std::array<float, 4> &orientation_quat,
    const std::array<float, 3> &angular_velocity_rad_s,
    const std::array<float, 3> &linear_acceleration_mps2) noexcept
{
    g_control_info.imu_orientation = orientation_quat;
    g_control_info.imu_angular_velocity = angular_velocity_rad_s;
    g_control_info.imu_linear_acceleration = linear_acceleration_mps2;
}

void BigRabbitControlBridge::SetSensorDataFootForce(const std::array<float, 2> &foot_force_n) noexcept
{
    g_control_info.foot_force_n = foot_force_n;
}

void BigRabbitControlBridge::SetSensorDataBaseHeight(float base_height_m) noexcept
{
    g_control_info.base_height_m = base_height_m;
}

void BigRabbitControlBridge::SetMotionCommand(const std::array<float, 3> &command) noexcept
{
    g_motion_command = command;
}

void BigRabbitControlBridge::SetActionHistory(
    const std::array<float, kJointAxisNum> &previous,
    const std::array<float, kJointAxisNum> &current) noexcept
{
    g_previous_action = previous;
    g_current_action = current;
}

void BigRabbitControlBridge::ExecuteJointSpaceControl() noexcept
{
    ExecuteJointSpaceControl(isaac_policy::kCrouchJointPositionRad);
}

void BigRabbitControlBridge::ExecuteJointSpaceControl(
    const std::array<float, kJointAxisNum> &joint_position_ref_rad) noexcept
{
    int param = 0;
    NextState result = NextState::OK;

    for (std::size_t i = 0; i < kJointAxisNum; i++)
    {
        g_control_info.joint_position_ref[i] = joint_position_ref_rad[i];
    }

    // 関節位置目標をモータ位置目標へ落とす。ここまでが 62.5 Hz。
    g_update_module->Invoke("", param, result);
    g_joint_position_transfer->Invoke("", param, result);
}

void BigRabbitControlBridge::ExecuteReferenceOnly(long long step_count) noexcept
{
    // action=0 の参照そのまま。PD（実機ではドライバ）の追従性能だけを見るための経路。
    const float phase_unit = isaac_policy::PhaseUnit(step_count);
    std::array<float, kJointAxisNum> reference_rad{};
    isaac_policy::ReferenceJointPosition(phase_unit, g_motion_command, reference_rad);

    g_debug_data.valid = true;
    g_debug_data.step_count = step_count;
    g_debug_data.phase_unit = phase_unit;
    g_debug_data.motion_command = g_motion_command;
    g_debug_data.reference_rad = reference_rad;
    g_debug_data.joint_position_ref = reference_rad;
    g_debug_data.next_action.fill(0.0f);

    ExecuteJointSpaceControl(reference_rad);
}

void BigRabbitControlBridge::ExecuteDriverStep() noexcept
{
    int param = 0;
    NextState result = NextState::OK;

    // ドライバは 1 kHz。motor_angle_ref は 62.5 Hz でしか更新されず、その間は保持される。
    g_update_module->Invoke("", param, result);
    g_motor_driver->Invoke("", param, result);
    g_torque_set->Invoke("", param, result);
    g_debug_data.driver_torque_ratio = g_motor_driver->torque_ratio();
}

void BigRabbitControlBridge::ExecuteRLControl(long long step_count, bool reset) noexcept
{
    // policy 1 step 分。直前に sim_main が SetSensorDataJoint() と
    // SetIsaacPolicySensorData() を呼んでいる前提で、最新センサから obs54 を作る。

    /*
    ◆追加モジュール検討
    ★pinocchioでやってみるか？
    ・ロボットモデル（２段階目）
        floating baseのロボットモデルを保持しておく
        set jointangleで各関節角度を入れる
        set jointvelocityで各関節速度を入れる
        胴体姿勢、並進位置はどうしておく？？
        set jointforce で関節トルクを入れる

    ・RobotUpdate（２段階目）
        ロボットモデルの更新

    ・EstimateContact（1段階目）
        ※まず実機実験を優先するため、最小構成で書く。
        トルク値から床反力を取り、しきい値以上だと接触とする
        ベース基準、足裏のヤコビ行列が取れればよい

    ・EstimateHeight（1段階目）
        ※まず実機実験を優先するため、最小構成で書く
        接触と関節角度、ベース角度から、ベースの高度を推定

    */

    if (reset)
    {
        g_previous_action.fill(0.0f);
        g_current_action.fill(0.0f);
    }

    // 歩容位相は自由走行クロック。**接地でリセットしない**。
    // 参照テーブルが不連続に飛ぶと、PD が数 m/s の足先指令として追ってしまう。
    const float phase_unit = isaac_policy::PhaseUnit(step_count);
    const float phase_rad = 2.0f * isaac_policy::kPi * phase_unit;
    const float gait_sin = std::sin(phase_rad);
    const float gait_cos = std::cos(phase_rad);

    // ---- センサ生値 -> observation の変換。ここが sim と実機で共通の経路 ----
    const auto projected_gravity = ProjectedGravityFromStore();
    const auto feet_contact = FeetContactFromStore();

    const auto &crouch = isaac_policy::kCrouchJointPositionRad;
    std::array<float, isaac_policy::kObsDim> obs{};

    static_assert(isaac_policy::kObsDim == 54);
    static_assert(isaac_policy::kActionDim == 10);

    // obs は index を明示して埋める。obs_index++ にすると並びのずれが見えなくなる。
    // 並びの正本は Isaac Lab の Active Observation Terms 出力。

    // obs[0]: base height [m]。センサが無いので外から与えられた値をそのまま使う。
    obs[0] = g_control_info.base_height_m;

    // obs[1..3]: projected gravity（body frame で見た world 重力方向）。clip は変換側で済み。
    obs[1] = projected_gravity[0];
    obs[2] = projected_gravity[1];
    obs[3] = projected_gravity[2];

    // obs[4..6]: base angular velocity [rad/s]。IMU ジャイロ（body frame）そのまま。
    obs[4] = g_control_info.imu_angular_velocity[0];
    obs[5] = g_control_info.imu_angular_velocity[1];
    obs[6] = g_control_info.imu_angular_velocity[2];

    // obs[7..8]: feet contact [left, right]
    obs[7] = feet_contact[0];
    obs[8] = feet_contact[1];

    // obs[9..11]: motion command [x, y, yaw]
    obs[9] = g_motion_command[0];
    obs[10] = g_motion_command[1];
    obs[11] = g_motion_command[2];

    // obs[12..13]: gait clock [sin, cos]
    obs[12] = gait_sin;
    obs[13] = gait_cos;

    // obs[14..23]: joint position relative to crouch [rad]
    for (std::size_t i = 0; i < kJointAxisNum; i++)
    {
        obs[14 + i] = g_control_info.joint_position[i] - crouch[i];
    }

    // obs[24..33]: joint velocity [rad/s]
    for (std::size_t i = 0; i < kJointAxisNum; i++)
    {
        obs[24 + i] = g_control_info.joint_velo[i];
    }

    // obs[34..43]: 2 step 前の action
    for (std::size_t i = 0; i < kJointAxisNum; i++)
    {
        obs[34 + i] = g_previous_action[i];
    }

    // obs[44..53]: 1 step 前の action
    for (std::size_t i = 0; i < kJointAxisNum; i++)
    {
        obs[44 + i] = g_current_action[i];
    }

    std::array<float, kJointAxisNum> next_action{};
    isaac_policy::Infer(obs.data(), next_action.data());

    // action は crouch からの正規化オフセットではなく、**位相参照からの**オフセット。
    //   joint_ref = q_ref(phi, command) + scale * action
    std::array<float, kJointAxisNum> reference_rad{};
    isaac_policy::ReferenceJointPosition(phase_unit, g_motion_command, reference_rad);

    std::array<float, kJointAxisNum> joint_ref{};
    for (std::size_t i = 0; i < kJointAxisNum; i++)
    {
        joint_ref[i] = reference_rad[i] + next_action[i] * isaac_policy::kJointActionScaleRad[i];
    }

    g_debug_data.valid = true;
    g_debug_data.step_count = step_count;
    g_debug_data.phase_unit = phase_unit;
    g_debug_data.gait_clock = {gait_sin, gait_cos};
    g_debug_data.motion_command = g_motion_command;
    g_debug_data.projected_gravity = projected_gravity;
    g_debug_data.feet_contact = feet_contact;
    g_debug_data.obs = obs;
    g_debug_data.previous_action = g_previous_action;
    g_debug_data.current_action = g_current_action;
    g_debug_data.next_action = next_action;
    g_debug_data.reference_rad = reference_rad;
    g_debug_data.joint_position_ref = joint_ref;

    // 次 step の action 履歴観測に備えて 1 step 進める。
    g_previous_action = g_current_action;
    g_current_action = next_action;

    ExecuteJointSpaceControl(joint_ref);
}

void BigRabbitControlBridge::GetMotorAngleRef(
    std::array<float, kJointAxisNum> &motor_angle_ref_rad) noexcept
{
    // これが実機の出力。ドライバが位置ループを持つので、位置目標を渡すだけでよい。
    motor_angle_ref_rad = g_control_info.motor_angle_ref;
}

void BigRabbitControlBridge::GetJointPositionRef(
    std::array<float, kJointAxisNum> &joint_position_ref_rad) noexcept
{
    joint_position_ref_rad = g_control_info.joint_position_ref;
}

void BigRabbitControlBridge::GetControlCommandsJoint(
    std::array<float, kJointAxisNum> &joint_torques_Nm_out) noexcept
{
    joint_torques_Nm_out = g_control_info.joint_torque_ref;
}

void BigRabbitControlBridge::GetMotorTorqueRef(
    std::array<float, kJointAxisNum> &motor_torques_Nm_out) noexcept
{
    motor_torques_Nm_out = g_control_info.motor_torque_ref;
}

std::array<float, 3> BigRabbitControlBridge::GetProjectedGravity() const noexcept
{
    return ProjectedGravityFromStore();
}

std::array<float, 2> BigRabbitControlBridge::GetFeetContact() const noexcept
{
    return FeetContactFromStore();
}

BigRabbitControlBridge::IsaacPolicyDebugData
BigRabbitControlBridge::GetIsaacPolicyDebugData() const noexcept
{
    return g_debug_data;
}
