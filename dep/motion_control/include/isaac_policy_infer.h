#pragma once

// 学習済み actor の推論。ランタイム依存を持たない手書き MLP。
// 実機 MCU に同じコードを載せる前提なので ONNX / TorchScript は使わない。
// rabbit_mujoco_2 の isaac_policy_infer.h と同じ構造（次元だけヘッダ側で決まる）。

#include <algorithm>
#include <cmath>

#include "isaac_policy_weights.h"

namespace isaac_policy
{

    inline float Elu(float value) noexcept
    {
        return value >= 0.0f ? value : std::exp(value) - 1.0f;
    }

    /// RSL-RL の EmpiricalNormalization と同じ式。
    inline void NormalizeObservation(const float *raw_obs, float *norm_obs) noexcept
    {
        for (int index = 0; index < kObsDim; ++index)
        {
            norm_obs[index] =
                (raw_obs[index] - kObsMean[index]) / (kObsStd[index] + kObsNormalizerEps);
        }
    }

    inline void Linear(const float *weight, const float *bias, const float *input,
                       float *output, int rows, int cols) noexcept
    {
        for (int row = 0; row < rows; ++row)
        {
            float sum = bias[row];
            for (int col = 0; col < cols; ++col)
            {
                sum += weight[row * cols + col] * input[col];
            }
            output[row] = sum;
        }
    }

    inline void LinearElu(const float *weight, const float *bias, const float *input,
                          float *output, int rows, int cols) noexcept
    {
        Linear(weight, bias, input, output, rows, cols);
        for (int row = 0; row < rows; ++row)
        {
            output[row] = Elu(output[row]);
        }
    }

    inline void Infer(const float *raw_obs, float *action) noexcept
    {
        float norm_obs[kObsDim]{};
        float hidden0[kHiddenDim]{};
        float hidden1[kHiddenDim]{};
        float hidden2[kHiddenDim]{};

        NormalizeObservation(raw_obs, norm_obs);
        LinearElu(kActor0Weight, kActor0Bias, norm_obs, hidden0, kHiddenDim, kObsDim);
        LinearElu(kActor2Weight, kActor2Bias, hidden0, hidden1, kHiddenDim, kHiddenDim);
        LinearElu(kActor4Weight, kActor4Bias, hidden1, hidden2, kHiddenDim, kHiddenDim);
        Linear(kActor6Weight, kActor6Bias, hidden2, action, kActionDim, kHiddenDim);
    }

}  // namespace isaac_policy
