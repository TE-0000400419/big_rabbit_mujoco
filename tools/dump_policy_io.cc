// クロス検証用。状態を与えて obs54 / action10 / 参照姿勢 / 関節目標を吐く。
//
// bridge の実経路（ExecuteRLControl）を通すので、obs の並び・参照テーブル補間・位相計算の
// どこがずれても検出できる。MuJoCo には依存しない。
//
// 入力は 1 行 1 ケース、空白区切りの 54 数値。**すべてセンサ生値**。
// bridge が生値から observation を作るので、quaternion -> 重力方向、法線力 -> 接地 0/1 の
// 変換もこの経路で検証できる。
//
//   step_count
//   base_height                （センサ無し。sim では真値）
//   imu_quaternion[4]          （w, x, y, z）
//   imu_angular_velocity[3]    （body frame）
//   foot_force[2]              （法線力 [N]。閾値化は bridge）
//   motion_command[3]
//   previous_action[10]
//   current_action[10]
//   joint_position[10]
//   joint_velocity[10]
//
// 出力は 1 行 1 ケース、CSV で obs54, action10, reference10, joint_ref10, phase。

#include <array>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "big_rabbit_control_bridge.h"

namespace
{
    constexpr std::size_t kJointNum = BigRabbitControlBridge::kJointAxisNum;
    constexpr std::size_t kInputCount = 1 + 1 + 4 + 3 + 2 + 3 + kJointNum * 4;
}

int main()
{
    BigRabbitControlBridge bridge;
    bridge.Initialize();
    bridge.ExecuteReady();

    // ヘッダ行。列名を出しておかないと突き合わせ側で取り違える。
    std::cout << "phase";
    for (int i = 0; i < isaac_policy::kObsDim; ++i) std::cout << ",obs" << i;
    for (std::size_t i = 0; i < kJointNum; ++i) std::cout << ",action" << i;
    for (std::size_t i = 0; i < kJointNum; ++i) std::cout << ",reference" << i;
    for (std::size_t i = 0; i < kJointNum; ++i) std::cout << ",joint_ref" << i;
    std::cout << "\n";
    std::cout << std::setprecision(9);

    std::string line;
    long long case_index = 0;
    while (std::getline(std::cin, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream stream(line);
        std::vector<double> values;
        double value = 0.0;
        while (stream >> value)
        {
            values.push_back(value);
        }
        if (values.size() != kInputCount)
        {
            std::cerr << "[FATAL] case " << case_index << ": 数値が " << values.size()
                      << " 個。" << kInputCount << " 個必要" << std::endl;
            return 1;
        }

        std::size_t cursor = 0;
        const long long step_count = static_cast<long long>(values[cursor++]);

        const float base_height = static_cast<float>(values[cursor++]);
        std::array<float, 4> imu_quat{};
        std::array<float, 3> imu_gyro{};
        std::array<float, 2> foot_force{};
        for (int i = 0; i < 4; ++i) imu_quat[i] = static_cast<float>(values[cursor++]);
        for (int i = 0; i < 3; ++i) imu_gyro[i] = static_cast<float>(values[cursor++]);
        for (int i = 0; i < 2; ++i) foot_force[i] = static_cast<float>(values[cursor++]);

        std::array<float, 3> command{};
        for (int i = 0; i < 3; ++i) command[i] = static_cast<float>(values[cursor++]);

        std::array<float, kJointNum> previous_action{};
        std::array<float, kJointNum> current_action{};
        std::array<float, kJointNum> joint_position{};
        std::array<float, kJointNum> joint_velocity{};
        std::array<float, kJointNum> joint_torque{};
        for (std::size_t i = 0; i < kJointNum; ++i) previous_action[i] = static_cast<float>(values[cursor++]);
        for (std::size_t i = 0; i < kJointNum; ++i) current_action[i] = static_cast<float>(values[cursor++]);
        for (std::size_t i = 0; i < kJointNum; ++i) joint_position[i] = static_cast<float>(values[cursor++]);
        for (std::size_t i = 0; i < kJointNum; ++i) joint_velocity[i] = static_cast<float>(values[cursor++]);

        bridge.SetSensorDataJoint(joint_position, joint_velocity, joint_torque);
        bridge.SetSensorDataIMU(imu_quat, imu_gyro, {0.0f, 0.0f, 0.0f});
        bridge.SetSensorDataFootForce(foot_force);
        bridge.SetSensorDataBaseHeight(base_height);
        bridge.SetMotionCommand(command);
        bridge.SetActionHistory(previous_action, current_action);
        bridge.ExecuteRLControl(step_count, /*reset=*/false);

        const auto debug = bridge.GetIsaacPolicyDebugData();
        std::cout << debug.phase_unit;
        for (int i = 0; i < isaac_policy::kObsDim; ++i) std::cout << "," << debug.obs[i];
        for (std::size_t i = 0; i < kJointNum; ++i) std::cout << "," << debug.next_action[i];
        for (std::size_t i = 0; i < kJointNum; ++i) std::cout << "," << debug.reference_rad[i];
        for (std::size_t i = 0; i < kJointNum; ++i) std::cout << "," << debug.joint_position_ref[i];
        std::cout << "\n";
        case_index++;
    }

    std::cerr << "[dump] " << case_index << " cases" << std::endl;
    return 0;
}
