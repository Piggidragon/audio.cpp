#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::modules {

struct MingPatchNetworkConfig {
    int64_t hidden_size = 0;
    int64_t heads = 0;
    int64_t depth = 0;
    int64_t intermediate_size = 0;
};

struct MingFlowConfig {
    MingPatchNetworkConfig network;
    MingPatchNetworkConfig aggregator;
    int64_t latent_dim = 0;
    int64_t conditioning_dim = 0;
    std::string network_prefix = "flowloss.cfm.model";
    std::string aggregator_prefix = "linear_proj_audio";
};

/** One generated latent patch and its aggregated LLM embedding, the input of the next decoder step. */
struct MingFlowPatch {
    std::vector<float> latent;
    std::vector<float> embedding;
};

class MingFlowRuntime {
public:
    MingFlowRuntime(std::shared_ptr<const assets::TensorSource> source,
        const MingFlowConfig & config, core::ExecutionContext & execution);
    ~MingFlowRuntime();

    std::vector<float> predict_velocity(const std::vector<float> & latent, const std::vector<float> & conditioning,
        const std::vector<float> & history, float time);

    std::vector<float> sample(const std::vector<float> & conditioning, const std::vector<float> & history,
        const std::vector<float> & noise, const std::vector<float> & sde_noise,
        float cfg = 2.0f, float sigma = 0.25f, float temperature = 0.0f);
    std::vector<float> aggregate(const std::vector<float> & latent);
    // sample() followed by aggregate() of its patch, run as one graph on the backend. Fewer than the official
    // ten Euler steps trade fidelity for speed.
    MingFlowPatch sample_and_aggregate(const std::vector<float> & conditioning, const std::vector<float> & history,
        const std::vector<float> & noise, const std::vector<float> & sde_noise,
        float cfg = 2.0f, float sigma = 0.25f, float temperature = 0.0f, int64_t steps = 10);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::modules
