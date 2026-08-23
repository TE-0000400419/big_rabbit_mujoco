#pragma once

// センサ生値 -> policy 観測 への変換。
//
// **この変換は sim と実機で完全に同一**なので bridge 側に置く。
// sim / 実機で違うのは「生値をどこから取るか」だけで、そこが sim_main と実機 HAL の境界になる。
//
// 変換を sim_main 側に置くと、実機でもう一度実装することになり、
// しかも 2 つの実装がずれても検出できない。

#include <algorithm>
#include <array>
#include <cmath>

namespace big_rabbit::sensor
{

    /// world のベクトルを body frame へ回す。quat は (w, x, y, z)。
    /// Isaac Lab の `quat_apply_inverse` と同じ（共役で回す）。
    inline std::array<float, 3> RotateWorldToBody(const std::array<float, 4> &quat_wxyz,
                                                  const std::array<float, 3> &vector_w) noexcept
    {
        double w = quat_wxyz[0];
        double x = -quat_wxyz[1];
        double y = -quat_wxyz[2];
        double z = -quat_wxyz[3];
        const double norm = std::sqrt(w * w + x * x + y * y + z * z);
        const double inv_norm = norm > 1.0e-12 ? 1.0 / norm : 1.0;
        w *= inv_norm;
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;

        const std::array<double, 3> v{vector_w[0], vector_w[1], vector_w[2]};
        const std::array<double, 3> t{
            2.0 * (y * v[2] - z * v[1]),
            2.0 * (z * v[0] - x * v[2]),
            2.0 * (x * v[1] - y * v[0]),
        };
        return {
            static_cast<float>(v[0] + w * t[0] + y * t[2] - z * t[1]),
            static_cast<float>(v[1] + w * t[1] + z * t[0] - x * t[2]),
            static_cast<float>(v[2] + w * t[2] + x * t[1] - y * t[0]),
        };
    }

    /// IMU の姿勢 quaternion から body frame の重力方向を作る。
    ///
    /// **加速度センサからは取らない。** 加速度センサ単体では機体自身の加速度と重力が
    /// 区別できないので歩行中は使えない。quaternion（ジャイロと加速度センサの融合結果）が正しい入力。
    inline std::array<float, 3> ProjectedGravityBody(const std::array<float, 4> &imu_quat_wxyz) noexcept
    {
        auto gravity = RotateWorldToBody(imu_quat_wxyz, {0.0f, 0.0f, -1.0f});
        // 学習側の observation は clip ±1 されている。
        for (auto &value : gravity)
        {
            value = std::clamp(value, -1.0f, 1.0f);
        }
        return gravity;
    }

    /// 足裏の法線力から接地 0/1 を作る。学習側の ContactSensor と同じ閾値を使う。
    inline float FootContactFromForce(float normal_force_n, float threshold_n) noexcept
    {
        return normal_force_n > threshold_n ? 1.0f : 0.0f;
    }

}  // namespace big_rabbit::sensor
