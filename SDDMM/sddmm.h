#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <cusparse.h>

#define WARP_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define PROFILE 200

// currently support vec_size=2
// template <int vec_size>
// __global__ void dvaltoval_kernel(
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr,
//     torch::PackedTensorAccessor32 < scalar_t, 2, torch::RestrictPtrTraits > dval,
//     torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> val,
//     const int nnz, const int n)
// {
//     // rowIdx = tid
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if(tid>n)
//         return;
//     int lb = __ldg(&rowPtr_bcsr[tid]);
//     int hb = __ldg(&rowPtr_bcsr[tid + 1]);
//     int rowIdx = tid * 2;
//     int cnt[2];
//     cnt[0] = __ldg(&rowPtr[rowIdx]);
//     cnt[1] = __ldg(&rowPtr[rowIdx + 1]);
//     scalar_t tmp;
// #pragma unroll
//     for (int i = 0; i < 2; i++)
//         for (; lb < hb; lb++)
//         {
//             tmp = 2dval [lb][i];
//             if (tmp != 0)
//                 val[cnt[i]++] = tmp;
//         }
// }

torch::Tensor genmask_cuda(
    torch::Tensor rowPtr_bcsr,
    torch::Tensor rowPtr,
    torch::Tensor blkval,
    int nnz);

template <int CF, int blkWarpNum>
torch::Tensor csr_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

template <int CF, int blkWarpNum>
torch::Tensor csr_split_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum>
torch::Tensor bcsr_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values);

template <int CF, int blkWarpNum>
torch::Tensor bcsr_split_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum, int vec_size>
torch::Tensor vec_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values);

// SVB (vec_size>2)
template <int CF, int blkWarpNum, int vec_size>
torch::Tensor vec_split_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum>
int csr_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

template <int CF, int blkWarpNum>
int csr_split_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum>
int bcsr_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz);

template <int CF, int blkWarpNum>
int bcsr_split_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz);

template <int CF, int blkWarpNum, int vec_size>
int vec_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz);

template <int CF, int blkWarpNum, int vec_size>
int vec_split_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz);

__global__ void warmup();

torch::Tensor cusparse_sddmm_compute(
    torch::Tensor A, // A_ M x k
    torch::Tensor B, // B_ transpose N x k
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor &values);

int cusparse_sddmm_profile(
    torch::Tensor A, // A_ M x k
    torch::Tensor B, // B_ transpose N x k
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor &values);
