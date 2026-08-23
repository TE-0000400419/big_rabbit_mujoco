#pragma once

// 位相依存の歩行参照。Isaac 側の PhaseReferenceJointPositionAction と同じ計算をする。
//
//   Rabbit:      joint_ref = crouch + scale * action
//   Big Rabbit:  joint_ref = q_ref(phi, command) + scale * action
//
// **関節目標そのものが位相の関数**なので、位相の扱いを間違えると関節指令が飛ぶ。
// v9 では参照テーブル自体に 0.03 m の不連続があり、PD が 2.9 m/s の足先指令として追って
// 転倒の主因になった（直すだけで転倒率 1.25% -> 0.32%）。
// 同じ理由で **接地による位相リセットはしない**。自由走行クロックが学習時と一致する唯一の形。

#include <algorithm>
#include <array>
#include <cmath>

#include "isaac_policy_config.h"
#include "isaac_walk_reference.h"

namespace isaac_policy
{

    /// 制御ステップ数から 0..1 の位相を作る。
    /// data->time ではなくステップ数を使う。学習側も episode_length_buf（整数）x step_dt。
    inline float PhaseUnit(long long step_count) noexcept
    {
        const float turns = static_cast<float>(step_count) * kStepDt * kGaitFrequencyHz;
        return turns - std::floor(turns);
    }

    /// 指令つきの参照姿勢 q_ref(phi, command) を作る。
    ///
    ///   amp = clamp(max(|v_x|, |yaw|), 0, 1)   指令ゼロなら 0 で crouch へ連続的に退化する
    ///   dir = v_x < -0.1 ? -1 : +1             後退は位相を逆回しする
    ///
    /// 参照を当てるのは hip_pitch / knee / ankle だけ。hip_roll / hip_yaw は crouch のまま
    /// （横方向のバランスは policy の担当）。
    inline void ReferenceJointPosition(float phase_unit,
                                       const std::array<float, 3> &motion_command,
                                       std::array<float, kActionDim> &reference_rad) noexcept
    {
        const float forward = motion_command[0];
        const float yaw = motion_command[2];
        const float amplitude = std::clamp(std::max(std::abs(forward), std::abs(yaw)), 0.0f, 1.0f);
        const float direction = forward < -0.1f ? -1.0f : 1.0f;

        float phase = phase_unit * direction;
        phase -= std::floor(phase);  // 逆回しでも 0..1 に収める

        const float position = phase * static_cast<float>(kReferenceSamples);
        const int base = static_cast<int>(std::floor(position));
        const float weight = position - std::floor(position);

        for (int i = 0; i < kActionDim; ++i)
        {
            const int column = kReferenceColumn[i];
            if (column < 0)
            {
                // 参照を当てない関節は crouch のまま。
                reference_rad[i] = kCrouchJointPositionRad[i];
                continue;
            }
            // 右脚は半周期ずらす。Isaac 側は torch.roll(series, shifts=N/2) で作っており、
            // roll の意味は right[i] = left[(i - N/2) mod N] なので符号を合わせる。
            // N/2 のときは +N/2 と -N/2 が一致するが、将来 shift を変えたときに壊れないようにする。
            const int shift = kReferenceIsRight[i] ? kReferenceHalfShift : 0;
            const int i0 = ((base - shift) % kReferenceSamples + kReferenceSamples) % kReferenceSamples;
            const int i1 = (i0 + 1) % kReferenceSamples;
            const float table =
                (1.0f - weight) * kReferenceTableRad[i0][column] + weight * kReferenceTableRad[i1][column];
            reference_rad[i] =
                kCrouchJointPositionRad[i] + amplitude * (table - kCrouchJointPositionRad[i]);
        }
    }

}  // namespace isaac_policy
