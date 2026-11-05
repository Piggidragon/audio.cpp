#include "mmq8s.cuh"
#include "mmq.cuh"

// Q8_0 stores a 2-byte scale in front of every 32 quants, so its rows are only 2-byte aligned and MMVQ reads
// them with 16-bit loads. Here each weight row holds its k int8 quants contiguously, followed by the k/32 fp16
// scales, so a warp streams a row with 16-byte loads (one 512-byte transaction per iteration).

static constexpr int Q8S_WARPS          = 8;   // one warp per row at a time
static constexpr int Q8S_ROWS_PER_BLOCK = 64;  // rows sharing one copy of the activations
static constexpr int Q8S_UNROLL         = 4;   // 512-byte chunks each warp loads ahead
static constexpr int Q8S_QUANT_WARPS    = 8;

// A single column (decode) shares little per block, and 64-row blocks leave the 6 SMs of a GTX 1050 Ti with too
// few blocks for matrices of ~1000 rows: smaller blocks are 7% faster over a Qwen2.5-0.5B layer (896 x 896,
// 1152 x 896, 9728 x 896, 896 x 4864) than MMVQ, the 64-row blocks 10% slower.
static constexpr int Q8S_WARPS_1          = 4;
static constexpr int Q8S_ROWS_PER_BLOCK_1 = 16;
static constexpr int Q8S_UNROLL_1         = 2;

// Activations become int8 with one scale per 32 values, as MMVQ quantizes them (q8_1 without the sum).
static __global__ void q8s_quantize(const float * __restrict__ x, int8_t * __restrict__ q, float * __restrict__ d,
        const int k) {
    const int col   = blockIdx.y;
    const int block = blockIdx.x * Q8S_QUANT_WARPS + threadIdx.y;
    if (block >= k / 32) {
        return;
    }
    const int   lane  = threadIdx.x;
    const float value = x[(int64_t) col * k + block * 32 + lane];
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, 32));
    }
    const float scale = amax / 127.0f;
    q[(int64_t) col * k + block * 32 + lane] = amax == 0.0f ? 0 : (int8_t) roundf(value / scale);
    if (lane == 0) {
        d[(int64_t) col * (k / 32) + block] = scale;
    }
}

// The activations of all columns are staged in shared memory once per block: read from L2 by every warp
// they would cost ncols times the weight traffic.
template <int ncols, int warps = Q8S_WARPS, int rows_per_block = Q8S_ROWS_PER_BLOCK, int unroll = Q8S_UNROLL>
static __global__ void __launch_bounds__(32 * warps) q8s_matvec(
        const char * __restrict__ w, const int64_t row_bytes, const int k,
        const int8_t * __restrict__ xq, const float * __restrict__ xd, const float * __restrict__ bias,
        float * __restrict__ dst, const int nrows, const bool round_bf16) {
    extern __shared__ int4 shared[];
    const int blocks = k / 32;
    int8_t * xs = (int8_t *) shared;
    float  * ds = (float *) (xs + ncols * k);
    const int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid; i < ncols * k / 16; i += 32 * warps) {
        shared[i] = ((const int4 *) xq)[i];
    }
    for (int i = tid; i < ncols * blocks; i += 32 * warps) {
        ds[i] = xd[i];
    }
    __syncthreads();

    const int lane = threadIdx.x;
    const int last = min(nrows, (int) (blockIdx.x + 1) * rows_per_block);
    for (int row = blockIdx.x * rows_per_block + threadIdx.y; row < last; row += warps) {
        const char * quants = w + row * row_bytes;
        const half * scales = (const half *) (quants + k);
        float acc[ncols] = {0.0f};
        for (int base = lane * 16; base < k; base += 32 * 16 * unroll) {
            // Issue all loads first so that several transactions per warp are in flight.
            int4  wq[unroll];
            float wd[unroll];
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                const int offset = base + u * 32 * 16;
                if (offset < k) {
                    wq[u] = *(const int4 *) (quants + offset);
                    wd[u] = __half2float(scales[offset / 32]);
                }
            }
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                const int offset = base + u * 32 * 16;
                if (offset >= k) {
                    break;
                }
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const int4 aq = *(const int4 *) (xs + j * k + offset);
                    int sum = ggml_cuda_dp4a(wq[u].x, aq.x, 0);
                    sum = ggml_cuda_dp4a(wq[u].y, aq.y, sum);
                    sum = ggml_cuda_dp4a(wq[u].z, aq.z, sum);
                    sum = ggml_cuda_dp4a(wq[u].w, aq.w, sum);
                    acc[j] += wd[u] * ds[j * blocks + offset / 32] * (float) sum;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            acc[j] = warp_reduce_sum<32>(acc[j]);
        }
        if (lane == 0) {
            const float b = bias ? bias[row] : 0.0f;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float value = acc[j] + b;
                if (round_bf16) {
                    value = __bfloat162float(__float2bfloat16(value));
                }
                dst[(int64_t) j * nrows + row] = value;
            }
        }
    }
}

// Above this many columns the weights go back to Q8_0 in a pool buffer for MMQ, which streams them once for all
// columns instead of once per 8 (prompt prefill).
static constexpr int Q8S_MAX_COLUMNS = 16;

// One block per row, assembled in shared memory so that global memory sees 4-byte loads and stores: Q8_0 blocks are
// only 2-byte aligned, rows of an even number of them 4-byte aligned.
static constexpr int Q8S_Q8_0_UNITS = sizeof(block_q8_0) / 2;  // 2-byte units: the scale, then 16 pairs of quants

static __global__ void q8s_to_q8_0(const char * __restrict__ w, const int64_t row_bytes, const int k,
        char * __restrict__ dst) {
    extern __shared__ uint16_t row_q8_0[];
    const int       blocks = k / 32;
    const char *    row    = w + blockIdx.x * row_bytes;
    for (int i = threadIdx.x; i < k / 4; i += blockDim.x) {
        const uint32_t quants = ((const uint32_t *) row)[i];
        uint16_t * unit = row_q8_0 + (i / 8) * Q8S_Q8_0_UNITS + 1 + (i % 8) * 2;
        unit[0] = quants & 0xffff;
        unit[1] = quants >> 16;
    }
    for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
        row_q8_0[i * Q8S_Q8_0_UNITS] = ((const uint16_t *) (row + k))[i];
    }
    __syncthreads();
    const int units = blocks * Q8S_Q8_0_UNITS;
    char * out = dst + (int64_t) blockIdx.x * units * 2;
    if (blocks % 2 == 0) {
        for (int i = threadIdx.x; i < units / 2; i += blockDim.x) {
            ((uint32_t *) out)[i] = ((const uint32_t *) row_q8_0)[i];
        }
    } else {
        for (int i = threadIdx.x; i < units; i += blockDim.x) {
            ((uint16_t *) out)[i] = row_q8_0[i];
        }
    }
}

static __global__ void q8s_bias_round(float * __restrict__ dst, const float * __restrict__ bias, const int nrows,
        const int64_t n, const bool round_bf16) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    float value = dst[index] + (bias ? bias[index % nrows] : 0.0f);
    if (round_bf16) {
        value = __bfloat162float(__float2bfloat16(value));
    }
    dst[index] = value;
}

static void mul_mat_q8s_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int k, const bool round_bf16) {
    const ggml_tensor * w     = dst->src[0];
    const ggml_tensor * x     = dst->src[1];
    const ggml_tensor * bias  = dst->src[2];
    const int64_t       nrows = w->ne[1];
    const int64_t       ncols = ggml_nrows(x);
    cudaStream_t stream = ctx.stream();

    // MMQ reads the last row up to a multiple of MATRIX_ROW_PADDING values: that padding must be zero.
    const int64_t nblocks = nrows * (k / 32);
    const size_t  padding = ggml_row_size(GGML_TYPE_Q8_0, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> q8_0(ctx.pool(), nblocks * sizeof(block_q8_0) + padding);
    CUDA_CHECK(cudaMemsetAsync(q8_0.get() + nblocks * sizeof(block_q8_0), 0, padding, stream));
    q8s_to_q8_0<<<nrows, 128, (k / 32) * sizeof(block_q8_0), stream>>>((const char *) w->data, w->nb[1], k,
        q8_0.get());

    ggml_tensor w_q8_0 = *w;  // keeps the weight buffer, whose usage tells MMQ not to clear padding
    w_q8_0.type     = GGML_TYPE_Q8_0;
    w_q8_0.data     = q8_0.get();
    w_q8_0.view_src = nullptr;
    w_q8_0.ne[0]    = k;
    w_q8_0.nb[0]    = sizeof(block_q8_0);
    w_q8_0.nb[1]    = (k / 32) * sizeof(block_q8_0);
    w_q8_0.nb[2]    = w_q8_0.nb[3] = nrows * w_q8_0.nb[1];

    ggml_tensor x_2d = *x;
    x_2d.ne[1] = ncols;
    x_2d.ne[2] = x_2d.ne[3] = 1;
    x_2d.nb[2] = x_2d.nb[3] = ncols * x->nb[1];

    ggml_tensor dst_2d = *dst;
    dst_2d.op = GGML_OP_MUL_MAT;
    memset(dst_2d.op_params, 0, sizeof(dst_2d.op_params));
    dst_2d.src[0] = &w_q8_0;
    dst_2d.src[1] = &x_2d;
    dst_2d.src[2] = nullptr;
    dst_2d.ne[1] = ncols;
    dst_2d.ne[2] = dst_2d.ne[3] = 1;
    dst_2d.nb[2] = dst_2d.nb[3] = ncols * dst->nb[1];
    ggml_cuda_mul_mat_q(ctx, &w_q8_0, &x_2d, nullptr, &dst_2d);

    if (bias || round_bf16) {
        const int64_t n = nrows * ncols;
        q8s_bias_round<<<(n + 255) / 256, 256, 0, stream>>>((float *) dst->data,
            bias ? (const float *) bias->data : nullptr, (int) nrows, n, round_bf16);
    }
}

void ggml_cuda_op_mul_mat_q8s(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * x    = dst->src[1];
    const ggml_tensor * bias = dst->src[2];
    const int  k          = ggml_get_op_params_i32(dst, 0);
    const bool round_bf16 = ggml_get_op_params_i32(dst, 1) != 0;
    const int  nrows      = (int) w->ne[1];
    const int  ncols      = (int) ggml_nrows(x);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst) && ncols >= 1);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (ncols > Q8S_MAX_COLUMNS && ggml_cuda_should_use_mmq(GGML_TYPE_Q8_0, cc, ncols, /*n_experts=*/0)) {
        mul_mat_q8s_mmq(ctx, dst, k, round_bf16);
        return;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int8_t> xq(ctx.pool(), (size_t) ncols * k);
    ggml_cuda_pool_alloc<float>  xd(ctx.pool(), (size_t) ncols * (k / 32));
    {
        const dim3 grid((k / 32 + Q8S_QUANT_WARPS - 1) / Q8S_QUANT_WARPS, ncols);
        q8s_quantize<<<grid, dim3(32, Q8S_QUANT_WARPS), 0, stream>>>((const float *) x->data, xq.get(), xd.get(), k);
    }
    const char * wdata = (const char *) w->data;
    const float * bdata = bias ? (const float *) bias->data : nullptr;
    if (ncols == 1) {
        q8s_matvec<1, Q8S_WARPS_1, Q8S_ROWS_PER_BLOCK_1, Q8S_UNROLL_1><<<(nrows + Q8S_ROWS_PER_BLOCK_1 - 1) /
            Q8S_ROWS_PER_BLOCK_1, dim3(32, Q8S_WARPS_1), (size_t) k * 9 / 8, stream>>>(wdata, w->nb[1], k, xq.get(),
            xd.get(), bdata, (float *) dst->data, nrows, round_bf16);
        return;
    }
    const dim3 grid((nrows + Q8S_ROWS_PER_BLOCK - 1) / Q8S_ROWS_PER_BLOCK);
    const dim3 block(32, Q8S_WARPS);
    // Larger batches run in groups of 8 columns, each streaming the weights once.
    for (int first = 0; first < ncols; first += 8) {
        const int8_t * gq = xq.get() + (int64_t) first * k;
        const float  * gd = xd.get() + (int64_t) first * (k / 32);
        float * ddata = (float *) dst->data + (int64_t) first * nrows;
        switch (std::min(8, ncols - first)) {
#define Q8S_CASE(n) case n: q8s_matvec<n><<<grid, block, (size_t) n * k * 9 / 8, stream>>>(wdata, w->nb[1], k, gq, gd, \
                bdata, ddata, nrows, round_bf16); break;
            Q8S_CASE(1) Q8S_CASE(2) Q8S_CASE(3) Q8S_CASE(4) Q8S_CASE(5) Q8S_CASE(6) Q8S_CASE(7) Q8S_CASE(8)
#undef Q8S_CASE
            default: GGML_ABORT("unreachable");
        }
    }
}

static __device__ __forceinline__ float rope_round_bf16(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

// One thread per pair; the operation order matches the unfused graph (mul, round, sub/add, round).
static __global__ void rope_pairs_bf16(const float * __restrict__ x, const float * __restrict__ cos,
        const float * __restrict__ sin, float * __restrict__ dst, const int64_t row_len, const int heads,
        const int half, const int positions, const int64_t pairs) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pairs) {
        return;
    }
    const int     pair = index % half;
    const int64_t head = index / half;  // over rows * heads
    const int64_t row  = head / heads;
    const float * in   = x + row * row_len + (head % heads) * 2 * half + 2 * pair;
    const int     slot = (row % positions) * half + pair;
    const float a = in[0];
    const float b = in[1];
    const float c = cos[slot];
    const float s = sin[slot];
    dst[2 * index]     = rope_round_bf16(rope_round_bf16(a * c) - rope_round_bf16(b * s));
    dst[2 * index + 1] = rope_round_bf16(rope_round_bf16(b * c) + rope_round_bf16(a * s));
}

void ggml_cuda_op_rope_pairs_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const int heads    = ggml_get_op_params_i32(dst, 0);
    const int head_dim = ggml_get_op_params_i32(dst, 1);
    const int half     = head_dim / 2;
    const int positions = (int) (ggml_nelements(dst->src[1]) / half);
    const int64_t pairs = ggml_nrows(x) * heads * half;
    const int block = 256;
    rope_pairs_bf16<<<(pairs + block - 1) / block, block, 0, ctx.stream()>>>((const float *) x->data,
        (const float *) dst->src[1]->data, (const float *) dst->src[2]->data, (float *) dst->data, x->ne[0], heads,
        half, positions, pairs);
}

// One block per row: sum of squares, then r(r(x * rsqrt(mean + eps)) * w).
static __global__ void rms_norm_bf16(const float * __restrict__ x, const float * __restrict__ w,
        float * __restrict__ dst, const int cols, const float eps) {
    const float * in  = x + (int64_t) blockIdx.x * cols;
    float       * out = dst + (int64_t) blockIdx.x * cols;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += in[i] * in[i];
    }
    __shared__ float partial[32];
    sum = warp_reduce_sum<32>(sum);
    if (threadIdx.x % 32 == 0) {
        partial[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0.0f;
        sum = warp_reduce_sum<32>(sum);
        if (threadIdx.x == 0) {
            partial[0] = sum;
        }
    }
    __syncthreads();
    const float scale = rsqrtf(partial[0] / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out[i] = rope_round_bf16(rope_round_bf16(scale * in[i]) * w[i]);
    }
}

void ggml_cuda_op_rms_norm_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    rms_norm_bf16<<<ggml_nrows(x), 256, 0, ctx.stream()>>>((const float *) x->data,
        (const float *) dst->src[1]->data, (float *) dst->data, (int) x->ne[0], eps);
}

// One warp per (row, head): the query attends to the rows of its group; lanes split the head dimension.
static __global__ void attn_rows_bf16(const char * __restrict__ q, const char * __restrict__ k,
        const char * __restrict__ v, float * __restrict__ dst, const int head_dim, const int heads, const int rows,
        const int group, const int64_t q1, const int64_t q2, const int64_t k1, const int64_t k2,
        const int64_t v1, const int64_t v2) {
    const int task = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (task >= rows * heads) {
        return;
    }
    const int lane  = threadIdx.x % 32;
    const int head  = task % heads;
    const int row   = task / heads;
    const int first = row - row % group;
    const float * query = (const float *) (q + row * q2 + head * q1);
    const float scale = rsqrtf((float) head_dim);
    float scores[32];
    float top = -INFINITY;
    for (int j = 0; j < group; ++j) {
        const float * key = (const float *) (k + (first + j) * k2 + head * k1);
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            dot += query[d] * key[d];
        }
        scores[j] = warp_reduce_sum<32>(dot) * scale;
        top = fmaxf(top, scores[j]);
    }
    float total = 0.0f;
    for (int j = 0; j < group; ++j) {
        scores[j] = expf(scores[j] - top);
        total += scores[j];
    }
    for (int d = lane; d < head_dim; d += 32) {
        float value = 0.0f;
        for (int j = 0; j < group; ++j) {
            value += scores[j] / total * ((const float *) (v + (first + j) * v2 + head * v1))[d];
        }
        dst[(int64_t) row * heads * head_dim + head * head_dim + d] = rope_round_bf16(value);
    }
}

void ggml_cuda_op_attn_rows_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const int head_dim = (int) q->ne[0];
    const int heads    = (int) q->ne[1];
    const int rows     = (int) q->ne[2];
    const int group    = ggml_get_op_params_i32(dst, 0);
    const int warps    = 4;
    const int tasks    = rows * heads;
    attn_rows_bf16<<<(tasks + warps - 1) / warps, 32 * warps, 0, ctx.stream()>>>((const char *) q->data,
        (const char *) k->data, (const char *) v->data, (float *) dst->data, head_dim, heads, rows, group,
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2]);
}
