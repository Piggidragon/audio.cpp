#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/runtime/streaming_audio.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::modules {

struct MingBackboneConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    int64_t attention_heads = 0;
    int64_t key_value_heads = 0;
    float rms_norm_eps = 0;
    float rope_theta = 0;
};

struct MingAudioEncoderConfig {
    MingBackboneConfig backbone;
    int64_t input_dim = 0;
    int64_t hop_size = 0;
    int64_t latent_dim = 0;
    int64_t sliding_window = 0;
    bool bf16_posterior = true;
};

struct MingAudioDecoderConfig {
    MingBackboneConfig backbone;
    int64_t latent_dim = 0;
    int64_t output_dim = 0;
    int64_t sliding_window = 0;
};

struct MingAudioCodecConfig {
    MingAudioEncoderConfig encoder;
    MingAudioDecoderConfig decoder;
    int64_t patch_size = 0;
    int sample_rate = 0;
    std::string encoder_prefix = "audio.encoder";
    std::string decoder_prefix = "audio.decoder";
};

class MingAudioEncoder {
public:
    MingAudioEncoder(std::shared_ptr<const assets::TensorSource> source,
        const MingAudioCodecConfig & config, core::ExecutionContext & execution,
        core::AttentionPreference attention = core::AttentionPreference::Auto);
    ~MingAudioEncoder();

    int64_t latent_frames(size_t samples) const;
    std::vector<float> encode(const std::vector<float> & waveform, const std::vector<float> & noise);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MingAudioDecoder {
public:
    MingAudioDecoder(std::shared_ptr<const assets::TensorSource> source,
        const MingAudioCodecConfig & config, core::ExecutionContext & execution,
        core::AttentionPreference attention = core::AttentionPreference::Auto);
    ~MingAudioDecoder();

    std::vector<float> decode(const std::vector<float> & latent);
    runtime::AudioBuffer decode_streaming_window(const runtime::StreamingAudioWindow<float> & window,
        const runtime::StreamingAudioConfig & config);

    // Incremental streaming: every frame passes the decoder stack once, with the keys and values of the
    // previous sliding window cached per layer, instead of re-decoding a left context window per event.
    // A patch is final once its successor is known (frames interpolate between neighbouring patches).
    void stream_reset();
    void stream_push(const std::vector<float> & patch);
    // Patches that can be decoded now; all pushed patches once the stream is finishing.
    size_t stream_ready(bool finishing) const;
    // Decodes the next `patches` ready patches and returns the audio that is complete; with `finishing`
    // and no patches left, the remaining tail is returned as well.
    std::vector<float> stream_decode(size_t patches, bool finishing);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::modules
