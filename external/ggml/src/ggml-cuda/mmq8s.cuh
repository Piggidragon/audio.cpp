#include "common.cuh"

// GGML_OP_MUL_MAT_Q8S: small-batch int8 matrix-vector product over 16-byte aligned rows with split scales.
void ggml_cuda_op_mul_mat_q8s(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_ROPE_PAIRS_BF16: the bf16-rounded rotation of interleaved pairs in one kernel.
void ggml_cuda_op_rope_pairs_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_RMS_NORM_BF16 and GGML_OP_ATTN_ROWS_BF16: the bf16-rounded norm and attention of small transformers.
void ggml_cuda_op_rms_norm_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_attn_rows_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
