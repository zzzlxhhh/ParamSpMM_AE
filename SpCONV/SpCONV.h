#pragma once
#include <vector>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
using namespace torch::indexing;

template <int CF, int blkWarpNum>
void csr_spmm_cuda(
    torch::Tensor &output,
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values);

template <int CF, int blkWarpNum>
void csr_split_spmm_cuda(
    torch::Tensor &output,
    torch::Tensor input,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum>
void bcsr_spmm_cuda(
    torch::Tensor &output,
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values);

template <int CF, int blkWarpNum>
void bcsr_split_spmm_cuda(
    torch::Tensor &output,
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row);

template <int CF, int blkWarpNum>
torch::Tensor csr_spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto HW = torch::mm(input, weight);
    auto output = torch::zeros_like(HW);
    csr_spmm_cuda<CF, blkWarpNum>(output, HW, rowPtr, colIdx, values);
    return output;
}

// update w and X
template <int CF, int blkWarpNum>
std::vector<torch::Tensor> csr_spmm_backward_cuda(
    torch::Tensor g_input,
    torch::Tensor H,
    torch::Tensor weight,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto AG = torch::zeros_like(g_input);
    csr_spmm_cuda<CF, blkWarpNum>(AG, g_input, rowPtr, colIdx, values);
    auto g_output = torch::mm(AG, weight.t());
    auto g_weight = torch::mm(H.t(), AG);
    return {g_output, g_weight};
}

template <int CF, int blkWarpNum>
torch::Tensor csr_split_spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    auto HW = torch::mm(input, weight);
    auto output = torch::zeros_like(HW);
    csr_split_spmm_cuda<CF, blkWarpNum>(output, HW, rowPtr_split, colIdx, values, target_row);
    return output;
}

template <int CF, int blkWarpNum>
std::vector<torch::Tensor> csr_split_spmm_backward_cuda(
    torch::Tensor g_input,
    torch::Tensor H,
    torch::Tensor weight,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    auto AG = torch::zeros_like(g_input);
    csr_split_spmm_cuda<CF, blkWarpNum>(AG, g_input, rowPtr_split, colIdx, values, target_row);
    auto g_output = torch::mm(AG, weight.t());
    auto g_weight = torch::mm(H.t(), AG);
    return {g_output, g_weight};
}

template <int CF, int blkWarpNum>
torch::Tensor bcsr_spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto HW = torch::mm(input, weight);
    const int padded = (HW.size(0)+2-1) / 2 * 2;
    auto output = torch::zeros({padded, HW.size(1)}, HW.options());
    bcsr_spmm_cuda<CF, blkWarpNum>(output, HW, rowPtr_bcsr, colIdx, values);
    return output.index({Slice(None,HW.size(0)),"..."});
    // auto HW = torch::mm(input, weight);
    // auto output = torch::zeros_like(HW);
    // bcsr_spmm_cuda<CF, blkWarpNum>(output, HW, rowPtr_bcsr, colIdx, values);
    // return output;
}

template <int CF, int blkWarpNum>
std::vector<torch::Tensor> bcsr_spmm_backward_cuda(
    torch::Tensor g_input,
    torch::Tensor H,
    torch::Tensor weight,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    // auto AG = torch::zeros_like(g_input);
    // bcsr_spmm_cuda<CF, blkWarpNum>(AG, g_input, rowPtr_bcsr, colIdx, values);
    // auto g_output = torch::mm(AG, weight.t());
    // auto g_weight = torch::mm(H.t(), AG);
    // return {g_output, g_weight};

    const int padded = (g_input.size(0)+2-1) / 2 * 2;
    auto AG = torch::zeros({padded, g_input.size(1)}, g_input.options());
    bcsr_spmm_cuda<CF, blkWarpNum>(AG, g_input, rowPtr_bcsr, colIdx, values);
    AG = AG.index({Slice(None,g_input.size(0)),"..."});
    auto g_output = torch::mm(AG, weight.t());
    auto g_weight = torch::mm(H.t(), AG);
    return {g_output, g_weight};
}

template <int CF, int blkWarpNum>
torch::Tensor bcsr_split_spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    // auto HW = torch::mm(input, weight);
    // auto output = torch::zeros_like(HW);
    // bcsr_split_spmm_cuda<CF, blkWarpNum>(output, HW, rowPtr_bcsr_split, colIdx, values, target_row);
    // return output;

    auto HW = torch::mm(input, weight);
    const int padded = (HW.size(0)+2-1) / 2 * 2;
    auto output = torch::zeros({padded, HW.size(1)}, HW.options());
    bcsr_split_spmm_cuda<CF, blkWarpNum>(output, HW, rowPtr_bcsr_split, colIdx, values, target_row);
    return output.index({Slice(None,HW.size(0)),"..."});
}

template <int CF, int blkWarpNum>
std::vector<torch::Tensor> bcsr_split_spmm_backward_cuda(
    torch::Tensor g_input,
    torch::Tensor H,
    torch::Tensor weight,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    // auto AG = torch::zeros_like(g_input);
    // bcsr_split_spmm_cuda<CF, blkWarpNum>(AG, g_input, rowPtr_bcsr_split, colIdx, values, target_row);
    // auto g_output = torch::mm(AG, weight.t());
    // auto g_weight = torch::mm(H.t(), AG);
    // return {g_output, g_weight};

    const int padded = (g_input.size(0)+2-1) / 2 * 2;
    auto AG = torch::zeros({padded, g_input.size(1)}, g_input.options());
    bcsr_split_spmm_cuda<CF, blkWarpNum>(AG, g_input, rowPtr_bcsr_split, colIdx, values, target_row);
    AG = AG.index({Slice(None,g_input.size(0)),"..."});
    auto g_output = torch::mm(AG, weight.t());
    auto g_weight = torch::mm(H.t(), AG);
    return {g_output, g_weight};
}
