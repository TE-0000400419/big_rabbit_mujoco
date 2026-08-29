#include "big_rabbit_control_bridge.h"

#include <algorithm>
#include <cmath>

#include "big_rabbit_robot_model.h"
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

// ---- 接地判定と骨盤高をどこから取るか ----
// **既定は実機相当の推定（ハードコードで ON）**。実機に載る経路をそのまま sim でも回す。
// sim の真値（MuJoCo の接触法線力と骨盤 world z）へ戻すのは、段 1-3 のクロス検証のように
// 「与えたセンサ生値から obs54 が作れるか」を見るときだけ。そのときは
// BIG_RABBIT_USE_SIM_TRUTH を定義する（make ESTIMATE=OFF / cmake -DBIG_RABBIT_SIM_TRUTH=ON）。
#ifndef BIG_RABBIT_USE_SIM_TRUTH
#define BIG_RABBIT_USE_STATE_ESTIMATION 1
#endif

namespace
{
    ControlInfo g_control_info;

    UpdateModule *g_update_module = nullptr;
    JointPositionTransfer *g_joint_position_transfer = nullptr;
    MotorDriverSim *g_motor_driver = nullptr;
    TorqueSet *g_torque_set = nullptr;

    BigRabbitControlBridge::IsaacPolicyDebugData g_debug_data;
    std::array<float, 3> g_motion_command{};

    // 接地とみなす推定床反力 [N]（骨盤基準 Z）。policy の閾値（kFootContactThresholdN = 1 N）は
    // 力センサ生値に対するもので、こちらはトルクからの推定なのでノイズ床が高い。別の値を持つ。
    //
    // 20 N の根拠（sim で MuJoCo の実接地と突き合わせた実測、11 s x 4 条件）:
    //   遊脚の推定値は平均 +2.7 N・最大 15.1 N、接地中は平均 62 N。
    //   閾値 20 N で一致率 98.4-100%（誤検出 0.00%、見逃し 1.3-1.6%）。
    //   15 N だと前進はわずかに良いが旋回で誤検出が出るので 20 N を採る。
    //   学習側 v24 が入れた接地ノイズ（miss 0.05 / false 0.02）より十分小さい。
    constexpr float kContactThresholdN = 20.0f;

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

// 共有ロボットモデルの更新（順運動学）
void BigRabbitControlBridge::ExecuteRobotUpdate() noexcept
{
    // 関節角は MJCF / Isaac / policy と同じ定義なので、センサ生値をそのまま渡せる
    // （モータ次元の符号・減速比は SetSensorDataMotor が既に戻している）。
    //
    // IMU の姿勢を渡すのは **骨盤高の逆算にだけ** 必要だから。
    // 接地判定は骨盤固定座標で完結するので姿勢は要らない。
    // 渡さない（nullptr）と重力基準の量は「直立」を仮定した値になる。
    BigRabbitRobotUpdate(g_control_info.joint_position.data(),
                         g_control_info.imu_orientation.data());
}

// 接触判定　簡易版
void BigRabbitControlBridge::EstimateContact(std::array<float, kJointAxisNum> &reference_rad, float &left_contact, float &right_contact)
{
    /*
    ・作業空間力と関節トルクの関係
    τ^T dq=F^T dx  dx/dt=J*dq/dt
    τ^T dq=F^T (J dq) -> J^T F = τ
    F = #J^T*τ 転置ヤコビ行列の疑似逆行列＊関節トルク

    ・転置ヤコビはどう作るか？
    ヤコビ行列の座標（作業空間の座標をどう取るかに依存）
    →まずはpevis骨盤をベースとする（胴体を固定する）

    ・幾何ヤコビ行列を構成する
    ※並進成分なのでどう作っても良いが、幾何ベースで作る
    Jgeom_trans=[pel_z1 x (pel_p_E - pel_p_1) , ... , pel_zn x (pel_p_E - pel_p_n)]
    →　骨盤基準各関節回転軸　pel_z_i
    　　骨盤基準接触フレーム位置　pel_p_E
    　　骨盤基準各関節位置　　pel_p_i

    ・骨盤固定座標基準Z方向成分が一定以上で接触とする


    */

    // ヤコビと各関節位置・回転軸は ExecuteRobotUpdate が作ってある（big_rabbit_robot_model.h）。
    const BigRabbitState &model = BigRabbitRobotState();

    // reference_rad は今は使わない。ドライバがトルクを返さない機体で
    // tau = Kp(q_ref - q) - Kd*dq として推定に使うための引数（実機で必要になったら使う）。
    (void)reference_rad;

    float *contact_out[BIG_RABBIT_LEG_NUM] = {&left_contact, &right_contact};
    for (int side = 0; side < BIG_RABBIT_LEG_NUM; side++)
    {
        const BigRabbitLegState &leg = model.state[side];

        // 脚 5 関節の実測トルク。sim では MuJoCo の qfrc_actuator、実機ではドライバの報告値。
        Eigen::Matrix<float, BIG_RABBIT_LEG_JOINT_NUM, 1> tau;
        for (int i = 0; i < BIG_RABBIT_LEG_JOINT_NUM; i++)
        {
            tau(i) = g_control_info.joint_torque[side * BIG_RABBIT_LEG_JOINT_NUM + i];
        }

        // 脚自身の重力トルク。これを入れないと推定値が実測から大きくずれる
        // （sim 実測: 入れると接地中 62.3 N で真値 63.5 N とほぼ一致、
        //   入れないと 51.0 N、遊脚も 0 でなく -10.1 N になる）。
        //   tau_g(i) = sum_j (z_i x (com_j - p_i)) . (m_j g)    g は骨盤基準の重力ベクトル
        Eigen::Matrix<float, BIG_RABBIT_LEG_JOINT_NUM, 1> tau_gravity;
        tau_gravity.setZero();
        const Vector3f gravity_vector = 9.81f * model.gravity_base; // 骨盤基準 [m/s^2]
        for (int i = 0; i < BIG_RABBIT_LEG_JOINT_NUM; i++)
        {
            for (int j = i; j < BIG_RABBIT_LEG_JOINT_NUM; j++) // i より先のリンクだけが乗る
            {
                const Vector3f arm = leg.com_base[j] - leg.p_base[i];
                const Vector3f weight = model.leg[side].l[j].m * gravity_vector;
                tau_gravity(i) += leg.z_base[i].cross(arm).dot(weight);
            }
        }

        // 骨盤を固定した準静的つり合い（仮想仕事）:
        //   tau . dq + sum_j (m_j g) . dp_com_j + F . dp_E = 0
        //   -> tau + tau_gravity + J^T F = 0
        //   -> F = -pinv(J^T) (tau + tau_gravity)          F = 床が足に与える力（床反力）
        // 並進 3 成分だけ見る（足裏まわりのモーメントは取らない）。
        // COD は特異点（膝が伸び切る等）でも破綻しない疑似逆。
        const Eigen::Matrix<float, 3, BIG_RABBIT_LEG_JOINT_NUM> jacobian = leg.J_sole.topRows<3>();
        const Vector3f ground_reaction =
            -jacobian.transpose().completeOrthogonalDecomposition().solve(tau + tau_gravity);

        *contact_out[side] =
            (ground_reaction.z() > kContactThresholdN) ? 1.0f : 0.0f;
    }
}

// 高度推定　簡易版
void BigRabbitControlBridge::EstimateHeight(float left_contact, float right_contact, float &base_height_m)
{
    /*
    ・骨盤高に対応するセンサは無いので、接地している足から逆算する。
    ・ExecuteRobotUpdate が、足裏カプセルの接地点を **重力基準**（原点は骨盤のまま、
      姿勢だけ IMU で回した座標）で作ってある。接地点は地面の上にあるので、
      その z は -（骨盤の高さ）になる。よって符号を反転すれば骨盤高。
    ・平地の前提。傾斜・段差では成り立たない。
    */

    const BigRabbitState &model = BigRabbitRobotState();
    const float contact[BIG_RABBIT_LEG_NUM] = {left_contact, right_contact};

    float lowest = 1.0e9f;
    bool any_contact = false;
    for (int side = 0; side < BIG_RABBIT_LEG_NUM; side++)
    {
        if (contact[side] < 0.5f)
        {
            continue;
        }
        any_contact = true;
        lowest = std::min(lowest, model.state[side].p_contact_a_grav.z());
        lowest = std::min(lowest, model.state[side].p_contact_b_grav.z());
    }

    if (!any_contact)
    {
        // 両脚とも遊脚（歩容の flight phase、または接地判定の見逃し）。
        // 一番低い点を接地しているとみなす。実際より低めに出るが破綻はしない。
        for (int side = 0; side < BIG_RABBIT_LEG_NUM; side++)
        {
            lowest = std::min(lowest, model.state[side].p_contact_a_grav.z());
            lowest = std::min(lowest, model.state[side].p_contact_b_grav.z());
        }
    }

    base_height_m = -lowest;
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

    // ---- 共有ロボットモデルを先に更新する ----
    // 接地判定（EstimateContact）も骨盤高（EstimateHeight）もこの結果を見るので、
    // obs54 を組み立てるより前に済ませておく。
    ExecuteRobotUpdate();

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

    // ---- 接地と骨盤高をどこから取るか（このファイル冒頭のマクロ定義を見る）----
    // この 2 つだけが sim と実機で供給元の違うセンサ。ほかは同じ経路を通る。
#ifdef BIG_RABBIT_USE_STATE_ESTIMATION
    std::array<float, 2> feet_contact{};
    // トルクの元になった位置目標（前回 step のもの）を渡す。
    EstimateContact(g_control_info.joint_position_ref, feet_contact[0], feet_contact[1]);
    float base_height_m = 0.0f;
    EstimateHeight(feet_contact[0], feet_contact[1], base_height_m);
#else
    const auto feet_contact = FeetContactFromStore();
    const float base_height_m = g_control_info.base_height_m;
#endif

    const auto &crouch = isaac_policy::kCrouchJointPositionRad;
    std::array<float, isaac_policy::kObsDim> obs{};

    static_assert(isaac_policy::kObsDim == 54);
    static_assert(isaac_policy::kActionDim == 10);

    // obs は index を明示して埋める。obs_index++ にすると並びのずれが見えなくなる。
    // 並びの正本は Isaac Lab の Active Observation Terms 出力。

    // obs[0]: base height [m]。sim は真値、実機相当では EstimateHeight の推定値。
    obs[0] = base_height_m;

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
