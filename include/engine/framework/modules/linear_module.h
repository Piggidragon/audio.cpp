#pragma once

#include "engine/framework/core/module.h"

#include <cstdint>
#include <optional>

namespace engine::modules {

struct LinearConfig {
    int64_t in_features = 0;
    int64_t out_features = 0;
    bool use_bias = true;
    ggml_prec precision = GGML_PREC_DEFAULT;
    bool cuda_nvfp4_f16_activation_lowering = false;
    bool vulkan_f32_activation_lowering = false;
    bool cuda_f16_output = false;
    // Round the output to bf16 (ggml_round_bf16), inside GGML_OP_MUL_MAT_Q8S for repacked Q8_0 weights.
    bool round_bf16_output = false;
};

struct LinearWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class LinearModule {
public:
    explicit LinearModule(LinearConfig config);

    const LinearConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const LinearWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    LinearConfig config_;
};

}  // namespace engine::modules
