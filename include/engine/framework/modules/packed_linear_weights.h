#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

struct PackedLinearSource {
    std::string weight;
    std::optional<std::string> bias;
    int64_t out_features = 0;
};

struct PackedLinearWeightsConfig {
    int64_t in_features = 0;
    std::vector<PackedLinearSource> sources;
    bool use_bias = false;
    bool defer_upload = false;
    // Q8_0 weights are stored repacked for GGML_OP_MUL_MAT_Q8S: an I8 tensor [out, in + in/16 padded to 16] whose
    // rows hold the int8 quants and then the fp16 scales. LinearModule recognizes the layout and uses that op. CUDA only.
    bool repack_q8s = false;
};

class PackedLinearWeightsBuilder {
public:
    explicit PackedLinearWeightsBuilder(PackedLinearWeightsConfig config);

    const PackedLinearWeightsConfig & config() const noexcept;

    LinearWeights build(
        core::BackendWeightStore & store,
        const assets::TensorSource & source,
        assets::TensorStorageType storage_type) const;

private:
    PackedLinearWeightsConfig config_;
};

}  // namespace engine::modules
