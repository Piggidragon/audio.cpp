#pragma once
// MMVQ launch parameters of the generic (NVIDIA) table: warps per block and rows per block by batch size.
// Batches of 5-8 columns use one warp and four rows per block: on a GTX 1050 Ti (sm_61) this is 20-30% faster
// than two warps and two rows (measured with test-backend-ops perf for Q8_0, 1024-4864 rows).
#define MMVQ_TUNE_NWARPS_1   4
#define MMVQ_TUNE_NWARPS_2_4 4
#define MMVQ_TUNE_NWARPS_5_8 1
#define MMVQ_TUNE_RPB_1      1
#define MMVQ_TUNE_RPB_2_4    2
#define MMVQ_TUNE_RPB_5_8    4
