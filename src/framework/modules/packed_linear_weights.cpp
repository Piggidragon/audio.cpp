#include "engine/framework/modules/packed_linear_weights.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

size_t row_bytes(ggml_type type, int64_t columns, const std::string & name) {
    const int64_t block_size = ggml_blck_size(type);
    if (block_size <= 0 || columns % block_size != 0) {
        throw std::runtime_error("Packed linear tensor is not block-aligned: " + name);
    }
    return ggml_row_size(type, columns);
}

}  // namespace

PackedLinearWeightsBuilder::PackedLinearWeightsBuilder(PackedLinearWeightsConfig config) : config_(std::move(config)) {
    if (config_.sources.empty()) {
        throw std::runtime_error("PackedLinearWeightsBuilder requires at least one source");
    }
    if (config_.in_features <= 0) {
        throw std::runtime_error("PackedLinearWeightsConfig.in_features must be positive");
    }
    for (const auto & source : config_.sources) {
        if (source.out_features <= 0) {
            throw std::runtime_error("PackedLinearSource.out_features must be positive");
        }
        if (config_.use_bias && !source.bias.has_value()) {
            throw std::runtime_error("Packed linear source is missing bias");
        }
    }
}

const PackedLinearWeightsConfig & PackedLinearWeightsBuilder::config() const noexcept {
    return config_;
}

LinearWeights PackedLinearWeightsBuilder::build(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType storage_type) const {
    const int64_t in_features = config_.in_features;
    int64_t out_features = 0;
    ggml_type packed_type = GGML_TYPE_COUNT;
    std::vector<std::byte> packed_weight;
    std::vector<std::byte> packed_bias;

    if (config_.defer_upload) {
        std::vector<core::BackendWeightStore::TensorPart> weight_parts;
        std::vector<core::BackendWeightStore::TensorPart> bias_parts;
        for (const auto & item : config_.sources) {
            weight_parts.push_back({item.weight, {item.out_features, in_features}});
            if (config_.use_bias) {
                bias_parts.push_back({*item.bias, {item.out_features}});
            }
        }
        LinearWeights weights;
        weights.weight = store.load_tensor(source, weight_parts, storage_type);
        if (config_.use_bias) {
            weights.bias = store.load_tensor(source, bias_parts, assets::TensorStorageType::F32);
        }
        return weights;
    }

    for (const auto & item : config_.sources) {
        const auto tensor = source.require_tensor(
            item.weight,
            storage_type,
            {item.out_features, in_features});
        if (packed_type == GGML_TYPE_COUNT) {
            packed_type = tensor.type;
        } else if (packed_type != tensor.type) {
            throw std::runtime_error("Packed linear sources resolved to different tensor types");
        }
        const size_t expected_bytes = row_bytes(tensor.type, in_features, item.weight) * item.out_features;
        if (tensor.bytes.size() != expected_bytes) {
            throw std::runtime_error("Packed linear tensor byte size mismatch: " + item.weight);
        }
        packed_weight.insert(packed_weight.end(), tensor.bytes.begin(), tensor.bytes.end());

        if (config_.use_bias) {
            const auto bias = source.require_tensor(*item.bias, assets::TensorStorageType::F32, {item.out_features});
            if (bias.type != GGML_TYPE_F32 || bias.bytes.size() != static_cast<size_t>(item.out_features) * sizeof(float)) {
                throw std::runtime_error("Packed linear bias must be F32: " + *item.bias);
            }
            packed_bias.insert(packed_bias.end(), bias.bytes.begin(), bias.bytes.end());
        }
        out_features += item.out_features;
    }

    LinearWeights weights;
    if (config_.repack_q8s && packed_type == GGML_TYPE_Q8_0 && in_features % 32 == 0) {
        // Q8_0 blocks are an fp16 scale and 32 int8 quants; split them into the quants and then the scales of a row.
        const size_t blocks = static_cast<size_t>(in_features / 32);
        const size_t q8s_row_bytes = static_cast<size_t>(GGML_PAD(in_features + in_features / 16, 16));
        std::vector<std::byte> repacked(static_cast<size_t>(out_features) * q8s_row_bytes);
        for (size_t row = 0; row < static_cast<size_t>(out_features); ++row) {
            std::byte * out = repacked.data() + row * q8s_row_bytes;
            for (size_t block = 0; block < blocks; ++block) {
                const std::byte * in = packed_weight.data() + (row * blocks + block) * 34;
                std::memcpy(out + block * 32, in + 2, 32);
                std::memcpy(out + in_features + block * 2, in, 2);
            }
        }
        weights.weight = store.make_tensor(
            core::TensorShape::from_dims({out_features, static_cast<int64_t>(q8s_row_bytes)}),
            GGML_TYPE_I8,
            repacked.data(),
            repacked.size());
    } else {
        weights.weight = store.make_tensor(
            core::TensorShape::from_dims({out_features, in_features}),
            packed_type,
            packed_weight.data(),
            packed_weight.size());
    }
    if (config_.use_bias) {
        weights.bias = store.make_tensor(
            core::TensorShape::from_dims({out_features}),
            GGML_TYPE_F32,
            packed_bias.data(),
            packed_bias.size());
    }
    return weights;
}

}  // namespace engine::modules
