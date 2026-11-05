#include "engine/framework/modules/speech_decoders/ming_flow_runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <array>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

std::vector<float> time_embedding(float time);

class MingPatchNetwork {
public:
    MingPatchNetwork(const assets::TensorSource & source, const MingPatchNetworkConfig & config,
        const std::string & prefix, int64_t latent_dim, int64_t llm_dim, bool conditioned,
        core::ExecutionContext & execution)
        : execution_(execution), store_(execution.backend(), execution.backend_type(), prefix + ".weights", 4 * 1024 * 1024),
          conditioned_(conditioned), latent_dim_(latent_dim), llm_dim_(llm_dim) {
        namespace binding = modules::binding;
        hidden_ = config.hidden_size;
        heads_ = config.heads;
        const int64_t depth = config.depth;
        intermediate_ = config.intermediate_size;
        const auto storage = assets::TensorStorageType::Native;
        input_projection_ = load_linear(source, {{prefix + ".x_embedder", hidden_}}, latent_dim_);
        final_norm_ = binding::norm_weight_from_source(store_, source, prefix + ".final_layer.norm_final", hidden_);
        output_dim_ = conditioned ? latent_dim_ : llm_dim_;
        output_projection_ = load_linear(source, {{prefix + ".final_layer.linear", output_dim_}}, hidden_);
        if (conditioned_) {
            condition_projection_ = load_linear(source, {{prefix + ".c_embedder.cond_embedder", hidden_}}, llm_dim_);
            time_in_ = load_linear(source, {{prefix + ".t_embedder.time_mlp.0", hidden_}}, 256);
            time_out_ = load_linear(source, {{prefix + ".t_embedder.time_mlp.2", hidden_}}, hidden_);
        } else {
            cls_ = store_.load_tensor(source, prefix + ".word_embedder.weight", storage, {1, hidden_});
        }
        auto frequencies = source.require_f32(prefix + ".rotary_embed.inv_freq");
        core::round_f32_to_bf16_in_place(frequencies);
        for (size_t i = 0; i < frequencies.size(); ++i) {
            frequencies[i] = std::pow(10000.0f, -static_cast<float>(2 * i) / (hidden_ / heads_)) / frequencies[i];
        }
        rope_factors_ = store_.make_f32(core::TensorShape::from_dims({hidden_ / heads_ / 2}), frequencies);
        for (int64_t i = 0; i < depth; ++i) {
            const auto p = prefix + ".blocks." + std::to_string(i);
            Block block;
            block.norm1 = binding::norm_weight_from_source(store_, source, p + ".norm1", hidden_);
            block.norm2 = binding::norm_weight_from_source(store_, source, p + ".norm2", hidden_);
            // One projection for q, k and v: one matmul launch, and q and k end up adjacent for a joint RoPE.
            block.qkv = load_linear(source, {{p + ".attn.to_q", hidden_}, {p + ".attn.to_k", hidden_},
                {p + ".attn.to_v", hidden_}}, hidden_);
            block.attn_out = load_linear(source, {{p + ".attn.to_out.0", hidden_}}, hidden_);
            block.ff_in = load_linear(source, {{p + ".mlp.ff.0.0", intermediate_}}, hidden_);
            block.ff_out = load_linear(source, {{p + ".mlp.ff.2", hidden_}}, intermediate_);
            blocks_.push_back(std::move(block));
        }
        store_.upload();
        graph_ = build(conditioned_ ? 2 : 1);
    }

    std::vector<float> run(const std::vector<float> & latent, const std::vector<float> & condition,
        const std::vector<float> & history, float time) {
        if (latent.empty() || latent.size() % latent_dim_ != 0 ||
            (conditioned_ && (latent.size() != static_cast<size_t>(latent_dim_) ||
                condition.size() != static_cast<size_t>(llm_dim_) || history.size() != latent.size()))) {
            throw std::runtime_error("Ming patch network input dimensions do not match the checkpoint");
        }
        const int64_t batch = conditioned_ ? 2 : latent.size() / latent_dim_;
        if (!conditioned_ && batch > 1 && (!reference_graph_ || reference_graph_->batch != batch)) {
            reference_graph_.reset();
            reference_graph_ = build(batch);
        }
        auto & graph = !conditioned_ && batch > 1 ? *reference_graph_ : *graph_;
        auto input = latent;
        if (conditioned_) input.insert(input.end(), latent.begin(), latent.end());
        core::write_tensor_f32(graph.input, input);
        if (conditioned_) {
            auto history_batch = history;
            history_batch.insert(history_batch.end(), history.begin(), history.end());
            auto condition_batch = condition;
            condition_batch.resize(2 * condition.size(), 0.0f);
            core::write_tensor_f32(graph.history, history_batch);
            core::write_tensor_f32(graph.condition, condition_batch);
            core::write_tensor_f32(graph.time, time_embedding(time));
        }
        if (core::compute_graph(execution_, graph.graph, graph.plan, "ming3_tts.patch") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Ming patch graph execution failed");
        }
        return core::read_tensor_f32(graph.output.tensor);
    }

private:
    /**
     * A biased projection followed by bf16 rounding, as every projection of the patch networks is. Q8_0 weights
     * are repacked for GGML_OP_MUL_MAT_Q8S (int8 rows with split fp16 scales), which also adds the bias and
     * rounds; other weights use the regular matmul.
     */
    struct Linear {
        int64_t in = 0, out = 0;
        core::TensorValue packed;
        std::optional<core::TensorValue> bias;
        modules::LinearWeights weights;
    };

    // `parts` are {prefix, out features}, stacked as rows; each prefix has .weight and .bias.
    Linear load_linear(const assets::TensorSource & source, const std::vector<std::pair<std::string, int64_t>> & parts,
        int64_t in) {
        Linear linear;
        linear.in = in;
        for (const auto & part : parts) linear.out += part.second;
        const size_t blocks = static_cast<size_t>(in / 32);
        const size_t row_bytes = static_cast<size_t>(in + in / 16);
        std::vector<std::byte> packed;
        std::vector<float> bias;
        bool q8 = in % 32 == 0;
        for (const auto & [prefix, rows] : parts) {
            if (!q8) break;
            const auto raw = source.require_tensor_data(prefix + ".weight");
            // Q8_0 blocks: an fp16 scale, then 32 int8 quants.
            if (raw.bytes.size() != static_cast<size_t>(rows) * blocks * 34) {
                q8 = false;
                break;
            }
            const size_t first = packed.size();
            packed.resize(first + static_cast<size_t>(rows) * row_bytes);
            for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
                std::byte * out = packed.data() + first + row * row_bytes;
                for (size_t block = 0; block < blocks; ++block) {
                    const std::byte * in_block = raw.bytes.data() + (row * blocks + block) * 34;
                    std::memcpy(out + block * 32, in_block + 2, 32);
                    std::memcpy(out + in + block * 2, in_block, 2);
                }
            }
            const auto values = source.require_f32(prefix + ".bias");
            bias.insert(bias.end(), values.begin(), values.end());
        }
        if (q8) {
            linear.packed = store_.make_tensor(core::TensorShape::from_dims({linear.out, static_cast<int64_t>(row_bytes)}),
                GGML_TYPE_I8, packed.data(), packed.size());
            linear.bias = store_.make_f32(core::TensorShape::from_dims({linear.out}), std::move(bias));
            return linear;
        }
        modules::PackedLinearWeightsConfig config{in, {}, true};
        for (const auto & [prefix, rows] : parts) config.sources.push_back({prefix + ".weight", prefix + ".bias", rows});
        linear.weights = modules::PackedLinearWeightsBuilder(config).build(store_, source, assets::TensorStorageType::Native);
        return linear;
    }

    core::TensorValue linear(core::ModuleBuildContext & ctx, const Linear & linear, const core::TensorValue & input) const {
        if (linear.packed.tensor != nullptr) {
            auto * x = ggml_is_contiguous(input.tensor) ? input.tensor : ggml_cont(ctx.ggml, input.tensor);
            auto shape = input.shape;
            shape.dims[shape.rank - 1] = linear.out;
            return core::wrap_tensor(ggml_mul_mat_q8s(ctx.ggml, linear.packed.tensor, x, linear.bias->tensor, true),
                shape, GGML_TYPE_F32);
        }
        return modules::Bf16RoundModule{}.build(ctx,
            modules::LinearModule({linear.in, linear.out, true}).build(ctx, input, linear.weights));
    }

    // RMS norm, rounded before and after the weight as the checkpoint's bf16 inference does; one fused op.
    core::TensorValue rms_norm(core::ModuleBuildContext & ctx, const core::TensorValue & x,
        const modules::NormWeights & weights) const {
        const auto & weight = *weights.weight;
        if (weight.type == GGML_TYPE_F32 && ggml_is_contiguous(x.tensor)) {
            return core::wrap_tensor(ggml_rms_norm_bf16(ctx.ggml, x.tensor, weight.tensor, 1.0e-6f), x.shape,
                GGML_TYPE_F32);
        }
        modules::NormConfig norm{hidden_, 1.0e-6f, true, false};
        norm.normalized_round_type = GGML_TYPE_BF16;
        return modules::Bf16RoundModule{}.build(ctx, modules::RMSNormModule(norm).build(ctx, x, weights));
    }

    struct Block {
        modules::NormWeights norm1, norm2;
        Linear qkv, attn_out, ff_in, ff_out;
    };

    struct Graph {
        Graph(core::ExecutionContext & execution, int64_t batch) : execution(execution), batch(batch) {}
        ~Graph() {
            core::release_backend_graph_resources(execution.backend(), graph, true);
            plan.reset();
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
        }
        core::ExecutionContext & execution;
        int64_t batch;
        core::TensorValue input, history, condition, time, positions, output;
        ggml_context * context = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t allocator = nullptr;
        core::HostGraphPlan plan;
    };

public:
    /** Conditioning shared by every flow step of one patch: projected history and LLM hidden state. */
    struct Prefix {
        core::TensorValue history, condition;
    };

    Prefix build_prefix(core::ModuleBuildContext & ctx, const core::TensorValue & history,
        const core::TensorValue & condition) const {
        return {linear(ctx, input_projection_, history), linear(ctx, condition_projection_, condition)};
    }

    /** Time embedding MLP; depends only on the step, not on the patch. */
    core::TensorValue build_time(core::ModuleBuildContext & ctx, const core::TensorValue & time) const {
        auto t = linear(ctx, time_in_, time);
        t = modules::SiluModule{}.build(ctx, t);
        t = modules::Bf16RoundModule{}.build(ctx, t);
        return linear(ctx, time_out_, t);
    }

    /** Rounded RoPE rotation table [1, half, 1, steps]: the same for every block and flow step of a graph. */
    struct RopeTable {
        ggml_tensor * cos = nullptr;
        ggml_tensor * sin = nullptr;
    };

    // As the rounded NEOX RoPEModule: rotate an identity vector, then round to bf16.
    RopeTable build_rope_table(core::ModuleBuildContext & ctx, const core::TensorValue & positions, int64_t steps) const {
        const int64_t dim = hidden_ / heads_;
        const int64_t half = dim / 2;
        auto * basis = ggml_new_tensor_4d(ctx.ggml, GGML_TYPE_F32, half, 1, steps, 1);
        auto * identity = ggml_concat(ctx.ggml, ggml_fill(ctx.ggml, basis, 1.0f), ggml_fill(ctx.ggml, basis, 0.0f), 0);
        modules::RoPEConfig rope{dim, GGML_ROPE_TYPE_NEOX, 10000.0f};
        auto table = modules::RoPEModule(rope).build(ctx,
            core::wrap_tensor(identity, core::TensorShape::from_dims({1, steps, 1, dim}), GGML_TYPE_F32),
            positions, &rope_factors_);
        auto * rounded = ggml_round_bf16(ctx.ggml, table.tensor);
        const auto half_view = [&](size_t offset) {
            auto * view = ggml_view_4d(ctx.ggml, rounded, half, 1, steps, 1, rounded->nb[1], rounded->nb[2],
                rounded->nb[3], offset);
            return ggml_reshape_4d(ctx.ggml, ggml_cont(ctx.ggml, view), 1, half, 1, steps);
        };
        return {half_view(0), half_view(half * sizeof(float))};
    }

    /** One conditioned velocity prediction; `input` is [batch, 1, latent], output [batch, 1, latent]. */
    core::TensorValue build_conditioned(core::ModuleBuildContext & ctx, const core::TensorValue & input,
        const Prefix & prefix, const core::TensorValue & time, const RopeTable & rope, int64_t batch) const {
        auto x = linear(ctx, input_projection_, input);
        auto t = modules::AddModule{}.build(ctx, time, prefix.condition);
        t = modules::Bf16RoundModule{}.build(ctx, t);
        x = modules::ConcatModule({1}).build(ctx, prefix.history, x);
        x = modules::ConcatModule({1}).build(ctx, t, x);
        return build_blocks(ctx, x, rope, batch, 3);
    }

    /** Aggregation of latent patches into LLM embeddings; `input` is [batch, 1, latent]. */
    core::TensorValue build_unconditioned(core::ModuleBuildContext & ctx, const core::TensorValue & input,
        const RopeTable & rope, int64_t batch) const {
        auto x = linear(ctx, input_projection_, input);
        const auto cls = core::wrap_tensor(ggml_cast(ctx.ggml, cls_.tensor, GGML_TYPE_F32), cls_.shape, GGML_TYPE_F32);
        auto cls_batch = ggml_repeat(ctx.ggml, ggml_reshape_3d(ctx.ggml, cls.tensor, hidden_, 1, 1), x.tensor);
        x = modules::ConcatModule({1}).build(ctx,
            core::wrap_tensor(cls_batch, core::TensorShape::from_dims({batch, 1, hidden_}), GGML_TYPE_F32), x);
        return build_blocks(ctx, x, rope, batch, 2);
    }

    int64_t latent_dim() const noexcept { return latent_dim_; }
    int64_t llm_dim() const noexcept { return llm_dim_; }

private:
    std::unique_ptr<Graph> build(int64_t batch) {
        auto graph = std::make_unique<Graph>(execution_, batch);
        const int64_t steps = conditioned_ ? 3 : 2;
        graph->context = ggml_init({4 * 1024 * 1024, nullptr, true});
        core::ModuleBuildContext ctx{graph->context, "ming3_tts.patch", execution_.backend_type()};
        graph->input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, latent_dim_}));
        ggml_set_input(graph->input.tensor);
        graph->positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        ggml_set_input(graph->positions.tensor);
        ggml_set_output(graph->positions.tensor);
        if (conditioned_) {
            graph->history = core::make_tensor(ctx, GGML_TYPE_F32, graph->input.shape);
            graph->condition = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, llm_dim_}));
            graph->time = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, 256}));
            ggml_set_input(graph->history.tensor); ggml_set_input(graph->condition.tensor); ggml_set_input(graph->time.tensor);
            graph->output = build_conditioned(ctx, graph->input, build_prefix(ctx, graph->history, graph->condition),
                build_time(ctx, graph->time), build_rope_table(ctx, graph->positions, steps), batch);
        } else {
            graph->output = build_unconditioned(ctx, graph->input, build_rope_table(ctx, graph->positions, steps), batch);
        }
        ggml_set_output(graph->output.tensor);
        graph->graph = ggml_new_graph_custom(graph->context, 8192, false);
        ggml_build_forward_expand(graph->graph, graph->output.tensor);
        graph->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!ggml_gallocr_reserve(graph->allocator, graph->graph) || !ggml_gallocr_alloc_graph(graph->allocator, graph->graph)) {
            throw std::runtime_error("Ming patch graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, graph->graph, graph->plan);
        const std::array<int32_t, 3> positions{0, 1, 2};
        ggml_backend_tensor_set(graph->positions.tensor, positions.data(), 0, steps * sizeof(int32_t));
        debug::timing_log_context_reservation("ming3_tts.patch", graph->context);
        return graph;
    }

    /** Transformer blocks, final norm and projection; returns the conditioned token or the CLS token. */
    core::TensorValue build_blocks(core::ModuleBuildContext & ctx, core::TensorValue x, const RopeTable & rope,
        int64_t batch, int64_t steps) const {
        const int64_t dim = hidden_ / heads_;
        for (const auto & block : blocks_) {
            auto normalized = rms_norm(ctx, x, block.norm1);
            auto * qkv = linear(ctx, block.qkv, normalized).tensor;
            const size_t row_bytes = qkv->nb[1];
            // The checkpoint rotates interleaved pairs (x[2i], x[2i+1]) with bf16 rounding after every product;
            // q and k are adjacent in each row, so one fused op covers both.
            auto * rotated = ggml_rope_pairs_bf16(ctx.ggml, qkv, rope.cos, rope.sin, static_cast<int>(2 * heads_),
                static_cast<int>(dim));
            // Attention among the tokens of each sequence, straight from the rotated q/k and the v of qkv.
            const auto heads_view = [&](ggml_tensor * source, size_t row, size_t offset) {
                return ggml_view_3d(ctx.ggml, source, dim, heads_, batch * steps, dim * sizeof(float), row, offset);
            };
            const size_t rotated_row = 2 * hidden_ * sizeof(float);
            auto * attended = ggml_attn_rows_bf16(ctx.ggml, heads_view(rotated, rotated_row, 0),
                heads_view(rotated, rotated_row, hidden_ * sizeof(float)),
                heads_view(qkv, row_bytes, 2 * hidden_ * sizeof(float)), static_cast<int>(steps));
            auto attention = core::wrap_tensor(ggml_reshape_3d(ctx.ggml, attended, hidden_, steps, batch),
                core::TensorShape::from_dims({batch, steps, hidden_}), GGML_TYPE_F32);
            attention = linear(ctx, block.attn_out, attention);
            x = modules::AddModule{}.build(ctx, x, attention);
            x = modules::Bf16RoundModule{}.build(ctx, x);
            normalized = rms_norm(ctx, x, block.norm2);
            auto ff = linear(ctx, block.ff_in, normalized);
            ff = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, ff);
            ff = modules::Bf16RoundModule{}.build(ctx, ff);
            ff = linear(ctx, block.ff_out, ff);
            x = modules::AddModule{}.build(ctx, x, ff);
            x = modules::Bf16RoundModule{}.build(ctx, x);
        }
        x = rms_norm(ctx, x, final_norm_);
        x = linear(ctx, output_projection_, x);
        auto output = modules::SliceModule({1, conditioned_ ? steps - 1 : 0, 1}).build(ctx, x);
        return core::wrap_tensor(ggml_cont(ctx.ggml, output.tensor), output.shape, output.type);
    }

    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    bool conditioned_;
    int64_t latent_dim_, llm_dim_, hidden_, heads_, intermediate_, output_dim_;
    std::vector<Block> blocks_;
    Linear input_projection_, output_projection_, condition_projection_, time_in_, time_out_;
    modules::NormWeights final_norm_;
    core::TensorValue cls_, rope_factors_;
    std::unique_ptr<Graph> graph_, reference_graph_;
};

/** The scaled sinusoidal time embedding of one flow step, as two identical rows for the guidance batch. */
std::vector<float> time_embedding(float time) {
    std::vector<float> scaled_time{1000.0f * time};
    core::round_f32_to_bf16_in_place(scaled_time);
    std::vector<float> embedding(512);
    const float frequency_step = static_cast<float>(std::log(10000.0) / 127.0);
    for (size_t i = 0; i < 128; ++i) {
        const float phase = scaled_time[0] * std::exp(-frequency_step * static_cast<float>(i));
        embedding[i] = embedding[256 + i] = std::sin(phase);
        embedding[128 + i] = embedding[384 + i] = std::cos(phase);
    }
    core::round_f32_to_bf16_in_place(embedding);
    return embedding;
}

constexpr size_t kFlowSteps = 10;
constexpr size_t kSamplerGraphSize = 32768;

/**
 * All ten Euler steps of one patch plus the aggregation of the result in a single graph: one upload and one
 * readback per patch instead of eleven graph runs with host round trips. Every op keeps the shapes and bf16
 * rounding of the step-by-step path, so the result is identical.
 */
struct MingFlowSampler {
    explicit MingFlowSampler(core::ExecutionContext & execution) : execution(execution) {}
    ~MingFlowSampler() {
        core::release_backend_graph_resources(execution.backend(), graph, true);
        plan.reset();
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
    }
    core::ExecutionContext & execution;
    float cfg = 0, sigma = 0, temperature = 0;
    size_t steps = 0;
    bool sde = false;
    core::TensorValue noise, history, condition, sde_noise, positions, aggregate_positions, times, latent, embedding;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    core::HostGraphPlan plan;
};

}  // namespace

struct MingFlowRuntime::Impl {
    Impl(const assets::TensorSource & source, const MingFlowConfig & config, core::ExecutionContext & execution)
        : execution(execution),
          flow(source, config.network, config.network_prefix,
              config.latent_dim, config.conditioning_dim, true, execution),
          aggregator(source, config.aggregator, config.aggregator_prefix,
              config.latent_dim, config.conditioning_dim, false, execution) {
        time = schedule(kFlowSteps);
        debug::trace_log_f32("ming3_tts.flow.timesteps", {11}, time);
    }

    // The official ten-step schedule; fewer steps sample its base curve at evenly spaced points.
    static std::vector<float> schedule(size_t steps) {
        static constexpr std::array<float, 11> base{0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
        std::vector<float> time(steps + 1);
        for (size_t i = 0; i <= steps; ++i) {
            const size_t scaled = i * 10;
            const size_t index = scaled / steps;
            const float fraction = static_cast<float>(scaled % steps) / static_cast<float>(steps);
            time[i] = scaled % steps == 0 ? base[index] : base[index] + fraction * (base[index + 1] - base[index]);
        }
        for (auto & t : time) t /= 32.0f;
        auto phase = time;
        for (auto & t : phase) t *= 1.5707963267948966f;
        core::round_f32_to_bf16_in_place(phase);
        for (auto & t : phase) t = std::cos(t);
        core::round_f32_to_bf16_in_place(phase);
        for (auto & t : phase) t -= 1.0f;
        core::round_f32_to_bf16_in_place(phase);
        for (size_t i = 0; i < time.size(); ++i) phase[i] += time[i];
        core::round_f32_to_bf16_in_place(phase);
        for (size_t i = 0; i < time.size(); ++i) time[i] -= phase[i];
        core::round_f32_to_bf16_in_place(time);
        return time;
    }

    std::unique_ptr<MingFlowSampler> build_sampler(float cfg, float sigma, float temperature, size_t steps) {
        auto sampler = std::make_unique<MingFlowSampler>(execution);
        sampler->steps = steps;
        const auto time = steps == kFlowSteps ? this->time : schedule(steps);
        sampler->cfg = cfg;
        sampler->sigma = sigma;
        sampler->temperature = temperature;
        // Zero noise adds nothing to an already rounded latent, so the SDE term only exists when it is nonzero.
        sampler->sde = sigma != 0.0f && temperature != 0.0f;
        const int64_t latent_dim = flow.latent_dim();
        const int64_t llm_dim = flow.llm_dim();
        sampler->context = ggml_init({ggml_tensor_overhead() * kSamplerGraphSize +
            ggml_graph_overhead_custom(kSamplerGraphSize, false), nullptr, true});
        core::ModuleBuildContext ctx{sampler->context, "ming3_tts.sampler", execution.backend_type()};
        const auto input = [&](ggml_type type, std::initializer_list<int64_t> dims, bool constant = false) {
            auto tensor = core::make_tensor(ctx, type, core::TensorShape::from_dims(dims));
            ggml_set_input(tensor.tensor);
            // Constants are written once; marking them as outputs keeps the allocator from reusing their memory.
            if (constant) ggml_set_output(tensor.tensor);
            return tensor;
        };
        sampler->noise = input(GGML_TYPE_F32, {latent_dim});
        sampler->history = input(GGML_TYPE_F32, {2, 1, latent_dim});
        sampler->condition = input(GGML_TYPE_F32, {2, 1, llm_dim});
        if (sampler->sde) sampler->sde_noise = input(GGML_TYPE_F32, {static_cast<int64_t>(steps), latent_dim});
        sampler->positions = input(GGML_TYPE_I32, {3}, true);
        sampler->aggregate_positions = input(GGML_TYPE_I32, {2}, true);
        sampler->times = input(GGML_TYPE_F32, {static_cast<int64_t>(steps), 2, 1, 256}, true);

        const auto prefix = flow.build_prefix(ctx, sampler->history, sampler->condition);
        const auto flow_rope = flow.build_rope_table(ctx, sampler->positions, 3);
        const size_t row_bytes = latent_dim * sizeof(float);
        ggml_tensor * y = sampler->noise.tensor;
        for (size_t step = 0; step < steps; ++step) {
            auto * times = sampler->times.tensor;
            const auto step_time = core::wrap_tensor(ggml_view_3d(ctx.ggml, times, 256, 1, 2, times->nb[1], times->nb[2],
                step * times->nb[3]), core::TensorShape::from_dims({2, 1, 256}), GGML_TYPE_F32);
            auto * row = ggml_reshape_3d(ctx.ggml, y, latent_dim, 1, 1);
            const auto batch = core::wrap_tensor(ggml_concat(ctx.ggml, row, row, 2),
                core::TensorShape::from_dims({2, 1, latent_dim}), GGML_TYPE_F32);
            const auto prediction = flow.build_conditioned(ctx, batch, prefix, flow.build_time(ctx, step_time),
                flow_rope, 2).tensor;
            auto * conditioned = ggml_view_1d(ctx.ggml, prediction, latent_dim, 0);
            auto * unconditioned = ggml_view_1d(ctx.ggml, prediction, latent_dim, row_bytes);
            std::vector<float> dt{time[step + 1] - time[step]};
            core::round_f32_to_bf16_in_place(dt);
            auto * guided = ggml_round_bf16(ctx.ggml, ggml_sub(ctx.ggml, conditioned, unconditioned));
            guided = ggml_round_bf16(ctx.ggml, ggml_scale(ctx.ggml, guided, cfg));
            guided = ggml_round_bf16(ctx.ggml, ggml_add(ctx.ggml, guided, conditioned));
            guided = ggml_round_bf16(ctx.ggml, ggml_scale(ctx.ggml, guided, dt[0]));
            y = ggml_round_bf16(ctx.ggml, ggml_add(ctx.ggml, y, guided));
            if (sampler->sde) {
                const float scale = sigma * std::sqrt(temperature) * std::sqrt(std::abs(dt[0]));
                auto * noise = ggml_view_1d(ctx.ggml, sampler->sde_noise.tensor, latent_dim, step * row_bytes);
                y = ggml_round_bf16(ctx.ggml, ggml_add(ctx.ggml, y, ggml_round_bf16(ctx.ggml, ggml_scale(ctx.ggml, noise, scale))));
            }
        }
        sampler->latent = core::wrap_tensor(y, core::TensorShape::from_dims({latent_dim}), GGML_TYPE_F32);
        ggml_set_output(y);
        const auto patch = core::wrap_tensor(ggml_reshape_3d(ctx.ggml, y, latent_dim, 1, 1),
            core::TensorShape::from_dims({1, 1, latent_dim}), GGML_TYPE_F32);
        sampler->embedding = aggregator.build_unconditioned(ctx, patch,
            aggregator.build_rope_table(ctx, sampler->aggregate_positions, 2), 1);
        ggml_set_output(sampler->embedding.tensor);

        sampler->graph = ggml_new_graph_custom(sampler->context, kSamplerGraphSize, false);
        ggml_build_forward_expand(sampler->graph, sampler->latent.tensor);
        ggml_build_forward_expand(sampler->graph, sampler->embedding.tensor);
        sampler->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (!ggml_gallocr_reserve(sampler->allocator, sampler->graph) ||
            !ggml_gallocr_alloc_graph(sampler->allocator, sampler->graph)) {
            throw std::runtime_error("Ming sampler graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, sampler->graph, sampler->plan);
        const std::array<int32_t, 3> positions{0, 1, 2};
        ggml_backend_tensor_set(sampler->positions.tensor, positions.data(), 0, 3 * sizeof(int32_t));
        ggml_backend_tensor_set(sampler->aggregate_positions.tensor, positions.data(), 0, 2 * sizeof(int32_t));
        std::vector<float> times;
        for (size_t step = 0; step < steps; ++step) {
            const auto embedding = time_embedding(time[step]);
            times.insert(times.end(), embedding.begin(), embedding.end());
        }
        core::write_tensor_f32(sampler->times, times);
        debug::timing_log_context_reservation("ming3_tts.sampler", sampler->context);
        return sampler;
    }

    MingFlowPatch sample(const std::vector<float> & conditioning, const std::vector<float> & history,
        const std::vector<float> & noise, const std::vector<float> & sde_noise, float cfg, float sigma, float temperature,
        size_t steps) {
        const int64_t latent_dim = flow.latent_dim();
        if (noise.size() != static_cast<size_t>(latent_dim) || history.size() != noise.size() ||
            conditioning.size() != static_cast<size_t>(flow.llm_dim())) {
            throw std::runtime_error("Ming sampler input dimensions do not match the checkpoint");
        }
        if (!sampler || sampler->cfg != cfg || sampler->sigma != sigma || sampler->temperature != temperature ||
            sampler->steps != steps) {
            sampler.reset();
            sampler = build_sampler(cfg, sigma, temperature, steps);
        }
        if (sampler->sde && sde_noise.size() < noise.size() * steps) {
            throw std::runtime_error("Ming flow requires ten SDE-noise frames");
        }
        core::write_tensor_f32(sampler->noise, noise);
        core::write_tensor_f32_slice(sampler->history, 0, history.data(), history.size());
        core::write_tensor_f32_slice(sampler->history, history.size(), history.data(), history.size());
        std::vector<float> condition(2 * conditioning.size(), 0.0f);
        std::copy(conditioning.begin(), conditioning.end(), condition.begin());
        core::write_tensor_f32(sampler->condition, condition);
        if (sampler->sde) core::write_tensor_f32(sampler->sde_noise, sde_noise.data(), noise.size() * steps);
        if (core::compute_graph(execution, sampler->graph, sampler->plan, "ming3_tts.sampler") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Ming sampler graph execution failed");
        }
        return {core::read_tensor_f32(sampler->latent.tensor), core::read_tensor_f32(sampler->embedding.tensor)};
    }

    core::ExecutionContext & execution;
    MingPatchNetwork flow, aggregator;
    std::vector<float> time;
    std::unique_ptr<MingFlowSampler> sampler;
};

MingFlowRuntime::MingFlowRuntime(std::shared_ptr<const assets::TensorSource> source,
    const MingFlowConfig & config, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(*source, config, execution)) {}
MingFlowRuntime::~MingFlowRuntime() = default;

std::vector<float> MingFlowRuntime::predict_velocity(const std::vector<float> & latent,
    const std::vector<float> & conditioning, const std::vector<float> & history, float time) {
    return impl_->flow.run(latent, conditioning, history, time);
}

std::vector<float> MingFlowRuntime::aggregate(const std::vector<float> & latent) {
    return impl_->aggregator.run(latent, {}, {}, 0.0f);
}

MingFlowPatch MingFlowRuntime::sample_and_aggregate(const std::vector<float> & conditioning,
    const std::vector<float> & history, const std::vector<float> & noise, const std::vector<float> & sde_noise,
    float cfg, float sigma, float temperature, int64_t steps) {
    if (temperature < 0) throw std::runtime_error("Ming flow requires a nonnegative temperature");
    if (steps < 1 || steps > 32) throw std::runtime_error("Ming flow steps must be between 1 and 32");
    return impl_->sample(conditioning, history, noise, sde_noise, cfg, sigma, temperature, static_cast<size_t>(steps));
}

std::vector<float> MingFlowRuntime::sample(const std::vector<float> & conditioning,
    const std::vector<float> & history, const std::vector<float> & noise, const std::vector<float> & sde_noise,
    float cfg, float sigma, float temperature) {
    if (sde_noise.size() != noise.size() * 10 || temperature < 0) {
        throw std::runtime_error("Ming flow requires ten SDE-noise frames and a nonnegative temperature");
    }
    const auto & time = impl_->time;
    auto y = noise;
    for (size_t step = 0; step < 10; ++step) {
        auto prediction = impl_->flow.run(y, conditioning, history, time[step]);
        std::vector<float> guided(noise.size());
        for (size_t i = 0; i < guided.size(); ++i) guided[i] = prediction[i] - prediction[guided.size() + i];
        core::round_f32_to_bf16_in_place(guided);
        for (auto & value : guided) value *= cfg;
        core::round_f32_to_bf16_in_place(guided);
        for (size_t i = 0; i < guided.size(); ++i) guided[i] += prediction[i];
        core::round_f32_to_bf16_in_place(guided);
        std::vector<float> dt{time[step + 1] - time[step]};
        core::round_f32_to_bf16_in_place(dt);
        for (auto & value : guided) value *= dt[0];
        core::round_f32_to_bf16_in_place(guided);
        for (size_t i = 0; i < y.size(); ++i) y[i] += guided[i];
        core::round_f32_to_bf16_in_place(y);
        for (size_t i = 0; i < y.size(); ++i) guided[i] = sigma * std::sqrt(temperature) *
            std::sqrt(std::abs(dt[0])) * sde_noise[step * y.size() + i];
        core::round_f32_to_bf16_in_place(guided);
        for (size_t i = 0; i < y.size(); ++i) y[i] += guided[i];
        core::round_f32_to_bf16_in_place(y);
    }
    return y;
}

}  // namespace engine::modules
