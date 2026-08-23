// 関節 <-> モータ変換の単体テスト。
// 符号を片方向だけ実装するミスをここで潰す。
#include <cmath>
#include <cstdio>

#include "joint_motor_transform.h"

namespace
{
    int failures = 0;

    void Check(bool ok, const char *what, double value)
    {
        std::printf("%s %-52s %.3e\n", ok ? "[ OK ]" : "[FAIL]", what, value);
        if (!ok) failures++;
    }
}

int main()
{
    using namespace big_rabbit;

    JointArray q{0.10f, -0.20f, 1.29f, -1.66f, 0.37f, -0.05f, 0.21f, 1.25f, -1.60f, 0.40f};

    // 1. joint -> motor -> joint が恒等
    MotorArray m{};
    JointArray back{};
    Joint2MotorPosition(q, m);
    Motor2JointPosition(m, back);
    double worst = 0.0;
    for (std::size_t i = 0; i < kJointNum; i++) worst = std::max(worst, std::abs(static_cast<double>(back[i] - q[i])));
    Check(worst < 1e-6, "joint -> motor -> joint 往復誤差", worst);

    // 2. トルクも往復で恒等
    JointArray tau_j{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -1.5f, 2.5f, -3.5f, 4.5f, -5.5f};
    MotorArray tau_m{};
    JointArray tau_back{};
    Joint2MotorTorque(tau_j, tau_m);
    Motor2JointTorque(tau_m, tau_back);
    worst = 0.0;
    for (std::size_t i = 0; i < kJointNum; i++) worst = std::max(worst, std::abs(static_cast<double>(tau_back[i] - tau_j[i])));
    Check(worst < 1e-4, "joint -> motor -> joint トルク往復誤差", worst);

    // 3. モータ次元 PD の関節等価剛性が Isaac の関節側ゲインと一致すること。
    //    符号は s^2 = 1 で打ち消えるので、方向補正に依存しないのが正しい挙動。
    worst = 0.0;
    const double dq_joint = 0.05;
    for (std::size_t i = 0; i < kJointNum; i++)
    {
        const double dq_motor = kJointDirection[i] * kGearRatio[i] * dq_joint;
        const double tau_motor = MotorStiffness(i) * dq_motor;
        const double tau_joint = kJointDirection[i] * kGearRatio[i] * tau_motor;
        const double expected = isaac_policy::kJointStiffness[i] * dq_joint;
        worst = std::max(worst, std::abs(tau_joint - expected));
    }
    Check(worst < 1e-5, "関節等価剛性が Isaac と一致（符号に依存しない）", worst);

    // 4. モータ側上限が GO / A1 でほぼ揃うこと（減速比と上限値の整合確認）
    double lo = 1e9, hi = -1e9;
    for (std::size_t i = 0; i < kJointNum; i++)
    {
        lo = std::min(lo, static_cast<double>(MotorEffortLimitNm(i)));
        hi = std::max(hi, static_cast<double>(MotorEffortLimitNm(i)));
    }
    Check(hi - lo < 0.1, "モータ側トルク上限のばらつき [N m]", hi - lo);
    std::printf("       モータ側トルク上限 %.3f .. %.3f N m\n", lo, hi);

    std::printf("\n%s (failures=%d)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
