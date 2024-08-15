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

template <int CF, int blkWarpNum>
torch::Tensor csr_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

template <int CF, int blkWarpNum>
torch::Tensor csr_split_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum>
torch::Tensor bcsr_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values);

template <int CF, int blkWarpNum>
torch::Tensor bcsr_split_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum, int vec_size>
torch::Tensor vec_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values);
// SVB (vec_size>2)
template <int CF, int blkWarpNum, int vec_size>
torch::Tensor vec_split_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum>
int csr_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

template <int CF, int blkWarpNum>
int csr_split_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum>
int bcsr_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz);

template <int CF, int blkWarpNum>
int bcsr_split_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz);

template <int CF, int blkWarpNum, int vec_size>
int vec_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz);

template <int CF, int blkWarpNum, int vec_size>
int vec_split_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz);

torch::Tensor cusparse_compute(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

int cusparse_profile(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

torch::Tensor cusparse_compute_csr_alg2(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

int cusparse_profile_csr_alg2(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

torch::Tensor cusparse_compute_coo_alg4(
    torch::Tensor input,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor values);

int cusparse_profile_coo_alg4(
    torch::Tensor input,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor values);

torch::Tensor cusparse_compute_csc(
    torch::Tensor input,
    torch::Tensor colPtr,
    torch::Tensor rowIdx,
    torch::Tensor values);

int cusparse_profile_csc(
    torch::Tensor input,
    torch::Tensor colPtr,
    torch::Tensor rowIdx,
    torch::Tensor values);

__global__ void warmup();

torch::Tensor gespmm_compute(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

int gespmm_profile(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);