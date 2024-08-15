#include <torch/extension.h>
#include <vector>
#include "SpMM.h"

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> split_csr(
    torch::Tensor rowPtr,
    int granularity)
{
  CHECK_CPU_INPUT(rowPtr);
  const int nrows = rowPtr.size(0) - 1;
  int *rowPtr_ac = rowPtr.data_ptr<int>(); // accessor
  std::vector<int> target_row;
  std::vector<int> rowPtr_split;
  rowPtr_split.push_back(0);
  for (int i = 0; i < nrows; i++)
  {
    int lb = rowPtr_ac[i];
    int hb = rowPtr_ac[i + 1];
    for (; lb < hb; lb += granularity)
    {
      target_row.push_back(i);
      if (lb + granularity > hb)
        rowPtr_split.push_back(hb);
      else
        rowPtr_split.push_back(lb + granularity);
    }
  }
  return {torch::from_blob(rowPtr_split.data(), {long(rowPtr_split.size())}, torch::kInt32).clone(),
          torch::from_blob(target_row.data(), {long(target_row.size())}, torch::kInt32).clone()};
}

std::vector<torch::Tensor> split_csr_par(
    torch::Tensor rowPtr,
    torch::Tensor row_split_offset,
    int granularity)
{
  CHECK_CPU_INPUT(rowPtr);
  CHECK_CPU_INPUT(row_split_offset);
  const int nrows = rowPtr.size(0) - 1;
  int *rowPtr_ac = rowPtr.data_ptr<int>();                     // accessor
  int *row_split_offset_ac = row_split_offset.data_ptr<int>(); // accessor
  const int split_nrows = row_split_offset_ac[nrows];

  std::vector<int> rowPtr_split(split_nrows + 1);
  std::vector<int> target_row(split_nrows);
  rowPtr_split[0] = 0;
#pragma omp parallel for
  for (int i = 0; i < nrows; i++)
  {
    int lb = rowPtr_ac[i];
    int hb = rowPtr_ac[i + 1];
    int offset = row_split_offset_ac[i];
    for (; lb < hb; lb += granularity)
    {
      target_row[offset] = i;
      if (lb + granularity > hb)
        rowPtr_split[offset + 1] = hb;
      else
        rowPtr_split[offset + 1] = lb + granularity;
      offset++;
    }
  }
  return {torch::from_blob(rowPtr_split.data(), {long(rowPtr_split.size())}, torch::kInt32).clone(),
          torch::from_blob(target_row.data(), {long(target_row.size())}, torch::kInt32).clone()};
}

torch::Tensor csr_spmm(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int csr_CF,
    int csr_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (csr_blkWarpNum == 4)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_spmm_cuda<1, 4>(input, rowPtr, colIdx, values);
    case 2:
      return csr_spmm_cuda<2, 4>(input, rowPtr, colIdx, values);
    case 3:
      return csr_spmm_cuda<3, 4>(input, rowPtr, colIdx, values);
    case 4:
      return csr_spmm_cuda<4, 4>(input, rowPtr, colIdx, values);
    case 5:
      return csr_spmm_cuda<5, 4>(input, rowPtr, colIdx, values);
    case 6:
      return csr_spmm_cuda<6, 4>(input, rowPtr, colIdx, values);
    case 7:
      return csr_spmm_cuda<7, 4>(input, rowPtr, colIdx, values);
    case 8:
      return csr_spmm_cuda<8, 4>(input, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (csr_blkWarpNum == 8)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_spmm_cuda<1, 8>(input, rowPtr, colIdx, values);
    case 2:
      return csr_spmm_cuda<2, 8>(input, rowPtr, colIdx, values);
    case 3:
      return csr_spmm_cuda<3, 8>(input, rowPtr, colIdx, values);
    case 4:
      return csr_spmm_cuda<4, 8>(input, rowPtr, colIdx, values);
    case 5:
      return csr_spmm_cuda<5, 8>(input, rowPtr, colIdx, values);
    case 6:
      return csr_spmm_cuda<6, 8>(input, rowPtr, colIdx, values);
    case 7:
      return csr_spmm_cuda<7, 8>(input, rowPtr, colIdx, values);
    case 8:
      return csr_spmm_cuda<8, 8>(input, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
  {
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
}

torch::Tensor csr_split_spmm(
    torch::Tensor input,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int csr_split_CF,
    int csr_split_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (csr_split_blkWarpNum == 4)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_spmm_cuda<1, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_spmm_cuda<2, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_spmm_cuda<3, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_spmm_cuda<4, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 5:
      return csr_split_spmm_cuda<5, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 6:
      return csr_split_spmm_cuda<6, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 7:
      return csr_split_spmm_cuda<7, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 8:
      return csr_split_spmm_cuda<8, 4>(input, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (csr_split_blkWarpNum == 8)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_spmm_cuda<1, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_spmm_cuda<2, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_spmm_cuda<3, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_spmm_cuda<4, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 5:
      return csr_split_spmm_cuda<5, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 6:
      return csr_split_spmm_cuda<6, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 7:
      return csr_split_spmm_cuda<7, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 8:
      return csr_split_spmm_cuda<8, 8>(input, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

torch::Tensor bcsr_spmm(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_bcsr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (bcsr_blkWarpNum == 4)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_spmm_cuda<1, 4>(input, rowPtr_bcsr, colIdx, values);
    case 2:
      return bcsr_spmm_cuda<2, 4>(input, rowPtr_bcsr, colIdx, values);
    case 3:
      return bcsr_spmm_cuda<3, 4>(input, rowPtr_bcsr, colIdx, values);
    case 4:
      return bcsr_spmm_cuda<4, 4>(input, rowPtr_bcsr, colIdx, values);
    case 5:
      return bcsr_spmm_cuda<5, 4>(input, rowPtr_bcsr, colIdx, values);
    case 6:
      return bcsr_spmm_cuda<6, 4>(input, rowPtr_bcsr, colIdx, values);
    case 7:
      return bcsr_spmm_cuda<7, 4>(input, rowPtr_bcsr, colIdx, values);
    case 8:
      return bcsr_spmm_cuda<8, 4>(input, rowPtr_bcsr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (bcsr_blkWarpNum == 8)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_spmm_cuda<1, 8>(input, rowPtr_bcsr, colIdx, values);
    case 2:
      return bcsr_spmm_cuda<2, 8>(input, rowPtr_bcsr, colIdx, values);
    case 3:
      return bcsr_spmm_cuda<3, 8>(input, rowPtr_bcsr, colIdx, values);
    case 4:
      return bcsr_spmm_cuda<4, 8>(input, rowPtr_bcsr, colIdx, values);
    case 5:
      return bcsr_spmm_cuda<5, 8>(input, rowPtr_bcsr, colIdx, values);
    case 6:
      return bcsr_spmm_cuda<6, 8>(input, rowPtr_bcsr, colIdx, values);
    case 7:
      return bcsr_spmm_cuda<7, 8>(input, rowPtr_bcsr, colIdx, values);
    case 8:
      return bcsr_spmm_cuda<8, 8>(input, rowPtr_bcsr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

torch::Tensor bcsr_split_spmm(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_bcsr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (bcsr_split_blkWarpNum == 4)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_spmm_cuda<1, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 2:
      return bcsr_split_spmm_cuda<2, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 3:
      return bcsr_split_spmm_cuda<3, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 4:
      return bcsr_split_spmm_cuda<4, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 5:
      return bcsr_split_spmm_cuda<5, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 6:
      return bcsr_split_spmm_cuda<6, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 7:
      return bcsr_split_spmm_cuda<7, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 8:
      return bcsr_split_spmm_cuda<8, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (bcsr_split_blkWarpNum == 8)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_spmm_cuda<1, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 2:
      return bcsr_split_spmm_cuda<2, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 3:
      return bcsr_split_spmm_cuda<3, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 4:
      return bcsr_split_spmm_cuda<4, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 5:
      return bcsr_split_spmm_cuda<5, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 6:
      return bcsr_split_spmm_cuda<6, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 7:
      return bcsr_split_spmm_cuda<7, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    case 8:
      return bcsr_split_spmm_cuda<8, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

torch::Tensor vec_spmm(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_bcsr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  const int vec_size = values.size(1);
  if (vec_size == 3)
  {
    if (bcsr_blkWarpNum == 4)
    {
      switch (bcsr_CF)
      {
      case 1:
        return vec_spmm_cuda<1, 4, 3>(input, rowPtr_bcsr, colIdx, values);
      case 2:
        return vec_spmm_cuda<2, 4, 3>(input, rowPtr_bcsr, colIdx, values);
      case 3:
        return vec_spmm_cuda<3, 4, 3>(input, rowPtr_bcsr, colIdx, values);
      case 4:
        return vec_spmm_cuda<4, 4, 3>(input, rowPtr_bcsr, colIdx, values);
      case 5:
        return vec_spmm_cuda<5, 4, 3>(input, rowPtr_bcsr, colIdx, values);
      case 6:
        return vec_spmm_cuda<6, 4, 3>(input, rowPtr_bcsr, colIdx, values);
      case 7:
        return vec_spmm_cuda<7, 4, 3>(input, rowPtr_bcsr, colIdx, values);
      case 8:
        return vec_spmm_cuda<8, 4, 3>(input, rowPtr_bcsr, colIdx, values);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else if (bcsr_blkWarpNum == 8)
    {
      switch (bcsr_CF)
      {
      case 1:
        return vec_spmm_cuda<1, 8, 3>(input, rowPtr_bcsr, colIdx, values);
      case 2:
        return vec_spmm_cuda<2, 8, 3>(input, rowPtr_bcsr, colIdx, values);
      case 3:
        return vec_spmm_cuda<3, 8, 3>(input, rowPtr_bcsr, colIdx, values);
      case 4:
        return vec_spmm_cuda<4, 8, 3>(input, rowPtr_bcsr, colIdx, values);
      case 5:
        return vec_spmm_cuda<5, 8, 3>(input, rowPtr_bcsr, colIdx, values);
      case 6:
        return vec_spmm_cuda<6, 8, 3>(input, rowPtr_bcsr, colIdx, values);
      case 7:
        return vec_spmm_cuda<7, 8, 3>(input, rowPtr_bcsr, colIdx, values);
      case 8:
        return vec_spmm_cuda<8, 8, 3>(input, rowPtr_bcsr, colIdx, values);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else
      TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
  else
    TORCH_CHECK(false, "vec_size must be 3");
}

torch::Tensor vec_split_spmm(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_bcsr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  const int vec_size = values.size(1);

  if (vec_size == 3)
  {
    if (bcsr_split_blkWarpNum == 4)
    {
      switch (bcsr_split_CF)
      {
      case 1:
        return vec_split_spmm_cuda<1, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 2:
        return vec_split_spmm_cuda<2, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 3:
        return vec_split_spmm_cuda<3, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 4:
        return vec_split_spmm_cuda<4, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 5:
        return vec_split_spmm_cuda<5, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 6:
        return vec_split_spmm_cuda<6, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 7:
        return vec_split_spmm_cuda<7, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 8:
        return vec_split_spmm_cuda<8, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else if (bcsr_split_blkWarpNum == 8)
    {
      switch (bcsr_split_CF)
      {
      case 1:
        return vec_split_spmm_cuda<1, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 2:
        return vec_split_spmm_cuda<2, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 3:
        return vec_split_spmm_cuda<3, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 4:
        return vec_split_spmm_cuda<4, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 5:
        return vec_split_spmm_cuda<5, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 6:
        return vec_split_spmm_cuda<6, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 7:
        return vec_split_spmm_cuda<7, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      case 8:
        return vec_split_spmm_cuda<8, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else
      TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
  else
    TORCH_CHECK(false, "vec_size must be 3");
}

int csr_spmm_profile(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int csr_CF,
    int csr_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (csr_blkWarpNum == 4)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_spmm_cuda_profile<1, 4>(input, rowPtr, colIdx, values);
    case 2:
      return csr_spmm_cuda_profile<2, 4>(input, rowPtr, colIdx, values);
    case 3:
      return csr_spmm_cuda_profile<3, 4>(input, rowPtr, colIdx, values);
    case 4:
      return csr_spmm_cuda_profile<4, 4>(input, rowPtr, colIdx, values);
    case 5:
      return csr_spmm_cuda_profile<5, 4>(input, rowPtr, colIdx, values);
    case 6:
      return csr_spmm_cuda_profile<6, 4>(input, rowPtr, colIdx, values);
    case 7:
      return csr_spmm_cuda_profile<7, 4>(input, rowPtr, colIdx, values);
    case 8:
      return csr_spmm_cuda_profile<8, 4>(input, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (csr_blkWarpNum == 8)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_spmm_cuda_profile<1, 8>(input, rowPtr, colIdx, values);
    case 2:
      return csr_spmm_cuda_profile<2, 8>(input, rowPtr, colIdx, values);
    case 3:
      return csr_spmm_cuda_profile<3, 8>(input, rowPtr, colIdx, values);
    case 4:
      return csr_spmm_cuda_profile<4, 8>(input, rowPtr, colIdx, values);
    case 5:
      return csr_spmm_cuda_profile<5, 8>(input, rowPtr, colIdx, values);
    case 6:
      return csr_spmm_cuda_profile<6, 8>(input, rowPtr, colIdx, values);
    case 7:
      return csr_spmm_cuda_profile<7, 8>(input, rowPtr, colIdx, values);
    case 8:
      return csr_spmm_cuda_profile<8, 8>(input, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
  {
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
}

int csr_split_spmm_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int csr_split_CF,
    int csr_split_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (csr_split_blkWarpNum == 4)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_spmm_cuda_profile<1, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_spmm_cuda_profile<2, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_spmm_cuda_profile<3, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_spmm_cuda_profile<4, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 5:
      return csr_split_spmm_cuda_profile<5, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 6:
      return csr_split_spmm_cuda_profile<6, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 7:
      return csr_split_spmm_cuda_profile<7, 4>(input, rowPtr_split, colIdx, values, target_row);
    case 8:
      return csr_split_spmm_cuda_profile<8, 4>(input, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (csr_split_blkWarpNum == 8)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_spmm_cuda_profile<1, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_spmm_cuda_profile<2, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_spmm_cuda_profile<3, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_spmm_cuda_profile<4, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 5:
      return csr_split_spmm_cuda_profile<5, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 6:
      return csr_split_spmm_cuda_profile<6, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 7:
      return csr_split_spmm_cuda_profile<7, 8>(input, rowPtr_split, colIdx, values, target_row);
    case 8:
      return csr_split_spmm_cuda_profile<8, 8>(input, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}
// special case when vec_size=2 float2 load could be used in kernel
int bcsr_spmm_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_bcsr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (bcsr_blkWarpNum == 4)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_spmm_cuda_profile<1, 4>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 2:
      return bcsr_spmm_cuda_profile<2, 4>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 3:
      return bcsr_spmm_cuda_profile<3, 4>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 4:
      return bcsr_spmm_cuda_profile<4, 4>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 5:
      return bcsr_spmm_cuda_profile<5, 4>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 6:
      return bcsr_spmm_cuda_profile<6, 4>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 7:
      return bcsr_spmm_cuda_profile<7, 4>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 8:
      return bcsr_spmm_cuda_profile<8, 4>(input, rowPtr_bcsr, colIdx, values, nnz);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (bcsr_blkWarpNum == 8)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_spmm_cuda_profile<1, 8>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 2:
      return bcsr_spmm_cuda_profile<2, 8>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 3:
      return bcsr_spmm_cuda_profile<3, 8>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 4:
      return bcsr_spmm_cuda_profile<4, 8>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 5:
      return bcsr_spmm_cuda_profile<5, 8>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 6:
      return bcsr_spmm_cuda_profile<6, 8>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 7:
      return bcsr_spmm_cuda_profile<7, 8>(input, rowPtr_bcsr, colIdx, values, nnz);
    case 8:
      return bcsr_spmm_cuda_profile<8, 8>(input, rowPtr_bcsr, colIdx, values, nnz);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}
// special case when vec_size=2 float2 load could be used in kernel
int bcsr_split_spmm_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_bcsr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (bcsr_split_blkWarpNum == 4)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_spmm_cuda_profile<1, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 2:
      return bcsr_split_spmm_cuda_profile<2, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 3:
      return bcsr_split_spmm_cuda_profile<3, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 4:
      return bcsr_split_spmm_cuda_profile<4, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 5:
      return bcsr_split_spmm_cuda_profile<5, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 6:
      return bcsr_split_spmm_cuda_profile<6, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 7:
      return bcsr_split_spmm_cuda_profile<7, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 8:
      return bcsr_split_spmm_cuda_profile<8, 4>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (bcsr_split_blkWarpNum == 8)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_spmm_cuda_profile<1, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 2:
      return bcsr_split_spmm_cuda_profile<2, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 3:
      return bcsr_split_spmm_cuda_profile<3, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 4:
      return bcsr_split_spmm_cuda_profile<4, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 5:
      return bcsr_split_spmm_cuda_profile<5, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 6:
      return bcsr_split_spmm_cuda_profile<6, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 7:
      return bcsr_split_spmm_cuda_profile<7, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 8:
      return bcsr_split_spmm_cuda_profile<8, 8>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

// specialized case of svb (vec_size=2)
int vec_spmm_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_bcsr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  const int vec_size = values.size(1);
  if (vec_size == 3)
  {
    if (bcsr_blkWarpNum == 4)
    {
      switch (bcsr_CF)
      {
      case 1:
        return vec_spmm_cuda_profile<1, 4, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 2:
        return vec_spmm_cuda_profile<2, 4, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 3:
        return vec_spmm_cuda_profile<3, 4, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 4:
        return vec_spmm_cuda_profile<4, 4, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 5:
        return vec_spmm_cuda_profile<5, 4, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 6:
        return vec_spmm_cuda_profile<6, 4, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 7:
        return vec_spmm_cuda_profile<7, 4, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 8:
        return vec_spmm_cuda_profile<8, 4, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else if (bcsr_blkWarpNum == 8)
    {
      switch (bcsr_CF)
      {
      case 1:
        return vec_spmm_cuda_profile<1, 8, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 2:
        return vec_spmm_cuda_profile<2, 8, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 3:
        return vec_spmm_cuda_profile<3, 8, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 4:
        return vec_spmm_cuda_profile<4, 8, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 5:
        return vec_spmm_cuda_profile<5, 8, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 6:
        return vec_spmm_cuda_profile<6, 8, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 7:
        return vec_spmm_cuda_profile<7, 8, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      case 8:
        return vec_spmm_cuda_profile<8, 8, 3>(input, rowPtr_bcsr, colIdx, values, nnz);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else
      TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
  else
    TORCH_CHECK(false, "vec_size must be 3");
}
// specialized case of svb (vec_size=2)
int vec_split_spmm_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(rowPtr_bcsr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  const int vec_size = values.size(1);
  if (vec_size == 3)
  {
    if (bcsr_split_blkWarpNum == 4)
    {
      switch (bcsr_split_CF)
      {
      case 1:
        return vec_split_spmm_cuda_profile<1, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 2:
        return vec_split_spmm_cuda_profile<2, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 3:
        return vec_split_spmm_cuda_profile<3, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 4:
        return vec_split_spmm_cuda_profile<4, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 5:
        return vec_split_spmm_cuda_profile<5, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 6:
        return vec_split_spmm_cuda_profile<6, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 7:
        return vec_split_spmm_cuda_profile<7, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 8:
        return vec_split_spmm_cuda_profile<8, 4, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else if (bcsr_split_blkWarpNum == 8)
    {
      switch (bcsr_split_CF)
      {
      case 1:
        return vec_split_spmm_cuda_profile<1, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 2:
        return vec_split_spmm_cuda_profile<2, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 3:
        return vec_split_spmm_cuda_profile<3, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 4:
        return vec_split_spmm_cuda_profile<4, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 5:
        return vec_split_spmm_cuda_profile<5, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 6:
        return vec_split_spmm_cuda_profile<6, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 7:
        return vec_split_spmm_cuda_profile<7, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 8:
        return vec_split_spmm_cuda_profile<8, 8, 3>(input, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else
      TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
  else
    TORCH_CHECK(false, "vec_size must be 3");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("split_csr", &split_csr, "split csr/bcsr with certain gran(CUDA)");
  m.def("split_csr_par", &split_csr_par, "split csr/bcsr with certain gran(CUDA) parallel");

  m.def("csr_spmm", &csr_spmm, "csr spmm (CUDA)");
  m.def("csr_split_spmm", &csr_split_spmm, "csr split spmm (CUDA)");

  m.def("bcsr_spmm", &bcsr_spmm, "bcsr spmm (CUDA)");
  m.def("bcsr_split_spmm", &bcsr_split_spmm, "bcsr split spmm (CUDA)");
  // vec_size>1
  m.def("vec_spmm", &vec_spmm, "bcsr spmm (CUDA)");
  m.def("vec_split_spmm", &vec_split_spmm, "bcsr split spmm (CUDA)");

  m.def("csr_spmm_profile", &csr_spmm_profile, "csr spmm profiling inside CPP ");
  m.def("csr_split_spmm_profile", &csr_split_spmm_profile, "csr split spmm profiling inside CPP ");
  // special case when vec_size=2 float2 load could be used in kernel
  m.def("bcsr_spmm_profile", &bcsr_spmm_profile, "bcsr spmm profiling inside CPP ");
  m.def("bcsr_split_spmm_profile", &bcsr_split_spmm_profile, "bcsr split spmm profiling inside CPP ");
  // vec_size>1
  m.def("vec_spmm_profile", &vec_spmm_profile, "vec spmm profiling inside CPP ");
  m.def("vec_split_spmm_profile", &vec_split_spmm_profile, "vec split spmm profiling inside CPP ");

  // vendor-provided cusparse by NVIDIA
  m.def("cusparse_compute", &cusparse_compute, "cusparse compute (CUDA)");
  m.def("cusparse_profile", &cusparse_profile, "cusparse compute (CUDA)");
  m.def("cusparse_compute_csr_alg2", &cusparse_compute_csr_alg2, "cusparse compute (CUDA)");
  m.def("cusparse_profile_csr_alg2", &cusparse_profile_csr_alg2, "cusparse compute (CUDA)");
  m.def("cusparse_compute_coo_alg4", &cusparse_compute_coo_alg4, "cusparse compute (CUDA)");
  m.def("cusparse_profile_coo_alg4", &cusparse_profile_coo_alg4, "cusparse compute (CUDA)");
  m.def("cusparse_compute_csc", &cusparse_compute_csc, "cusparse compute (CUDA)");
  m.def("cusparse_profile_csc", &cusparse_profile_csc, "cusparse compute (CUDA)");
  // sota static SpMM by GeSpMM
  m.def("gespmm_compute", &gespmm_compute, "gespmm compute (CUDA)");
  m.def("gespmm_profile", &gespmm_profile, "gespmm compute (CUDA)");
}