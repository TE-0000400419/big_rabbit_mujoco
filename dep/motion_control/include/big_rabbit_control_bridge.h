#pragma once

// Big Rabbit の制御 bridge。**センサ生値と指令を受け、モータ位置目標を出す**のが役割。
//
// rabbit_mujoco_2 の RabbitControlBridge に対応するが、Big Rabbit は位置制御ループを
// ドライバが持つので、加速度制御と外乱オブザーバの 2 段が無い。
//
// ## sim と実機の境界
//
// bridge に入るのは**すべてセンサ生値**にしてある。
// センサ生値 -> observation の変換（quaternion -> 重力方向、法線力 -> 接地 0/1）は
// sim と実機で完全に同一なので bridge が持つ（`sensor_transform.h`）。
// 差し替えるのは「生値をどこから取るか」だけ。
//
//   sim  : src/sim_main.cc が MuJoCo から生値を作る
//   実機 : USB コントローラの HAL が生値を作る
//
// ## 入出力の次元
//
//   実機の入力はモータ次元（USB コントローラからモータ角・速度・トルクが来る）
//     -> SetSensorDataMotor()
//   sim の入力は関節次元（MuJoCo の qpos/qvel）
//     -> SetSensorDataJoint()
//   どちらも内部で両方の次元へ展開される。
//
//   実機の出力はモータ位置目標（ドライバが位置ループを持つ）
//     -> GetMotorAngleRef()
//   sim の出力は関節トルク（MuJoCo の actuator へ渡す）
//     -> GetControlCommandsJoint()
//
// ## 実行周期
//
//   ExecuteRLControl()  62.5 Hz  obs54 -> 推論 -> joint_ref -> motor_angle_ref（保持）
//   ExecuteDriverStep()  1 kHz   motor_angle_ref -> motor_torque_ref -> joint_torque_ref
//                                （**sim 専用**。実機ではドライバのファームが担う）

#include <array>
#include <cstddef>

#include "isaac_policy_config.h"

class BigRabbitControlBridge
{
public:
    static constexpr std::size_t kJointAxisNum = 10;

    /// 追跡・比較用に、policy へ入れた値と出した値を丸ごと保持する。
    struct IsaacPolicyDebugData
    {
        bool valid = false;
        long long step_count = 0;
        float phase_unit = 0.0f;
        std::array<float, 2> gait_clock{};
        std::array<float, 3> motion_command{};
        /// bridge が quaternion から計算した重力方向。sim_main 側の値ではない。
        std::array<float, 3> projected_gravity{};
        /// bridge が法線力から閾値化した接地。
        std::array<float, 2> feet_contact{};
        std::array<float, isaac_policy::kObsDim> obs{};
        std::array<float, kJointAxisNum> previous_action{};
        std::array<float, kJointAxisNum> current_action{};
        std::array<float, kJointAxisNum> next_action{};
        std::array<float, kJointAxisNum> reference_rad{};
        std::array<float, kJointAxisNum> joint_position_ref{};
        float driver_torque_ratio = 0.0f;
    };

    void Initialize() noexcept;
    void ExecuteReady() noexcept;

    // ---- センサ生値の入力 ----

    /// 関節次元のセンサ。sim（MuJoCo の qpos/qvel）で使う。
    void SetSensorDataJoint(const std::array<float, kJointAxisNum> &joint_positions_rad,
                            const std::array<float, kJointAxisNum> &joint_velocities_rad_s,
                            const std::array<float, kJointAxisNum> &joint_torques_Nm) noexcept;

    /// モータ次元のセンサ。**実機（USB コントローラ）で使う**。
    /// 内部で符号 x 減速比を戻して関節次元へ展開する。
    void SetSensorDataMotor(const std::array<float, kJointAxisNum> &motor_positions_rad,
                            const std::array<float, kJointAxisNum> &motor_velocities_rad_s,
                            const std::array<float, kJointAxisNum> &motor_torques_Nm) noexcept;

    /// IMU の生値。姿勢 quaternion は (w, x, y, z)。
    /// 重力方向は bridge が quaternion から計算する。**加速度センサからは取らない**
    /// （加速度センサ単体では機体の加速度と重力が区別できず歩行中は使えない）。
    /// 角速度は body frame。observation の `base_ang_vel` にそのまま入る。
    void SetSensorDataIMU(const std::array<float, 4> &orientation_quat,
                          const std::array<float, 3> &angular_velocity_rad_s,
                          const std::array<float, 3> &linear_acceleration_mps2) noexcept;

    /// 足裏の法線力 [N]。0/1 への閾値化は bridge が行う
    /// （学習側と同じ閾値を 1 箇所で持つため）。
    void SetSensorDataFootForce(const std::array<float, 2> &foot_force_n) noexcept;

    /// 骨盤高 [m]。対応するセンサが無いので、sim では真値を入れる。
    /// 実機では支持脚から運動学で逆算した値を入れる（未実装）。
    void SetSensorDataBaseHeight(float base_height_m) noexcept;

    /// 移動指令 (x, y, yaw)。実行中に変えてよい。
    void SetMotionCommand(const std::array<float, 3> &command) noexcept;

    /// action 履歴を外から与える。**クロス検証専用**。実機の制御経路では使わない。
    void SetActionHistory(const std::array<float, kJointAxisNum> &previous,
                          const std::array<float, kJointAxisNum> &current) noexcept;

    // ---- 制御 ----
    /// crouch 保持。起動直後と debug 用。
    void ExecuteJointSpaceControl() noexcept;
    /// 指定した関節位置目標を保持する。
    void ExecuteJointSpaceControl(const std::array<float, kJointAxisNum> &joint_position_ref_rad) noexcept;
    /// policy 1 step 分。62.5 Hz で呼ぶ。
    void ExecuteRLControl(long long step_count, bool reset) noexcept;
    /// policy を通さず、参照姿勢そのまま（action=0）を関節目標にする。追従性能の測定用。
    void ExecuteReferenceOnly(long long step_count) noexcept;
    /// モータドライバ 1 step 分。1 kHz で呼ぶ。**sim 専用**。
    void ExecuteDriverStep() noexcept;

    // ---- 出力 ----

    /// モータ位置目標 [rad]。**これが実機の出力**。USB コントローラへ渡す。
    void GetMotorAngleRef(std::array<float, kJointAxisNum> &motor_angle_ref_rad) noexcept;
    /// 関節位置目標 [rad]。参考値（モータ次元に落とす前）。
    void GetJointPositionRef(std::array<float, kJointAxisNum> &joint_position_ref_rad) noexcept;
    /// 関節側トルク [N m]。**sim 専用**。MuJoCo の actuator へ渡す。
    void GetControlCommandsJoint(std::array<float, kJointAxisNum> &joint_torques_Nm_out) noexcept;
    /// ドライバが出したモータ側トルク [N m]。飽和の解析用。**sim 専用**。
    void GetMotorTorqueRef(std::array<float, kJointAxisNum> &motor_torques_Nm_out) noexcept;

    /// 現在のセンサ生値から作った重力方向。RL を回していない状態でも読める（ログ用）。
    std::array<float, 3> GetProjectedGravity() const noexcept;
    /// 現在のセンサ生値から作った接地 0/1。同上。
    std::array<float, 2> GetFeetContact() const noexcept;

    IsaacPolicyDebugData GetIsaacPolicyDebugData() const noexcept;

    // ---- 状態推定 ----

    /// 共有ロボットモデル（big_rabbit_robot_model.h）を最新のセンサ生値で更新する。
    /// 順運動学を回し、骨盤基準の各関節位置・回転軸・足裏位置・足裏ヤコビ、
    /// および重力基準の接地点を作る。ExecuteRLControl が obs を作る前に呼ぶ。
    /// 結果は BigRabbitRobotState() でどこからでも読める。
    void ExecuteRobotUpdate() noexcept;

    /// 接地判定（実機相当）。足裏力センサを使わず、関節トルクと足裏ヤコビから床反力を推定する。
    /// ExecuteRobotUpdate を先に呼んでおくこと。
    void EstimateContact(std::array<float, kJointAxisNum> &reference_rad, float &left_contact, float &right_contact);

    /// 骨盤高の推定（実機相当）。対応センサが無いので、接地脚の足裏位置から逆算する。
    /// ExecuteRobotUpdate を先に呼んでおくこと。
    void EstimateHeight(float left_contact, float right_contact, float &base_height_m);
};
