#include <torch/extension.h>
#include "SpCONV.h"

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

torch::Tensor csr_spmm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int csr_CF,
    int csr_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(rowPtr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (csr_blkWarpNum == 4)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_spmm_forward_cuda<1, 4>(input, weight, rowPtr, colIdx, values);
    case 2:
      return csr_spmm_forward_cuda<2, 4>(input, weight, rowPtr, colIdx, values);
    case 3:
      return csr_spmm_forward_cuda<3, 4>(input, weight, rowPtr, colIdx, values);
    case 4:
      return csr_spmm_forward_cuda<4, 4>(input, weight, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else if (csr_blkWarpNum == 8)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_spmm_forward_cuda<1, 8>(input, weight, rowPtr, colIdx, values);
    case 2:
      return csr_spmm_forward_cuda<2, 8>(input, weight, rowPtr, colIdx, values);
    case 3:
      return csr_spmm_forward_cuda<3, 8>(input, weight, rowPtr, colIdx, values);
    case 4:
      return csr_spmm_forward_cuda<4, 8>(input, weight, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else
  {
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
}

std::vector<torch::Tensor> csr_spmm_backward(
    torch::Tensor g_input,
    torch::Tensor H,
    torch::Tensor weight,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int csr_CF,
    int csr_blkWarpNum)
{
  CHECK_INPUT(g_input);
  CHECK_INPUT(H);
  CHECK_INPUT(weight);
  CHECK_INPUT(rowPtr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (csr_blkWarpNum == 4)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_spmm_backward_cuda<1, 4>(g_input, H, weight, rowPtr, colIdx, values);
    case 2:
      return csr_spmm_backward_cuda<2, 4>(g_input, H, weight, rowPtr, colIdx, values);
    case 3:
      return csr_spmm_backward_cuda<3, 4>(g_input, H, weight, rowPtr, colIdx, values);
    case 4:
      return csr_spmm_backward_cuda<4, 4>(g_input, H, weight, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else if (csr_blkWarpNum == 8)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_spmm_backward_cuda<1, 8>(g_input, H, weight, rowPtr, colIdx, values);
    case 2:
      return csr_spmm_backward_cuda<2, 8>(g_input, H, weight, rowPtr, colIdx, values);
    case 3:
      return csr_spmm_backward_cuda<3, 8>(g_input, H, weight, rowPtr, colIdx, values);
    case 4:
      return csr_spmm_backward_cuda<4, 8>(g_input, H, weight, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else
  {
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
}

torch::Tensor csr_split_spmm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int csr_split_CF,
    int csr_split_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(rowPtr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (csr_split_blkWarpNum == 4)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_spmm_forward_cuda<1, 4>(input, weight, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_spmm_forward_cuda<2, 4>(input, weight, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_spmm_forward_cuda<3, 4>(input, weight, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_spmm_forward_cuda<4, 4>(input, weight, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else if (csr_split_blkWarpNum == 8)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_spmm_forward_cuda<1, 8>(input, weight, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_spmm_forward_cuda<2, 8>(input, weight, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_spmm_forward_cuda<3, 8>(input, weight, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_spmm_forward_cuda<4, 8>(input, weight, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

std::vector<torch::Tensor> csr_split_spmm_backward(
    torch::Tensor g_input,
    torch::Tensor H,
    torch::Tensor weight,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int csr_split_CF,
    int csr_split_blkWarpNum)
{
  CHECK_INPUT(g_input);
  CHECK_INPUT(H);
  CHECK_INPUT(weight);
  CHECK_INPUT(rowPtr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (csr_split_blkWarpNum == 4)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_spmm_backward_cuda<1, 4>(g_input, H, weight, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_spmm_backward_cuda<2, 4>(g_input, H, weight, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_spmm_backward_cuda<3, 4>(g_input, H, weight, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_spmm_backward_cuda<4, 4>(g_input, H, weight, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else if (csr_split_blkWarpNum == 8)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_spmm_backward_cuda<1, 8>(g_input, H, weight, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_spmm_backward_cuda<2, 8>(g_input, H, weight, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_spmm_backward_cuda<3, 8>(g_input, H, weight, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_spmm_backward_cuda<4, 8>(g_input, H, weight, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

torch::Tensor bcsr_spmm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(rowPtr_bcsr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (bcsr_blkWarpNum == 4)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_spmm_forward_cuda<1, 4>(input, weight, rowPtr_bcsr, colIdx, values);
    case 2:
      return bcsr_spmm_forward_cuda<2, 4>(input, weight, rowPtr_bcsr, colIdx, values);
    case 3:
      return bcsr_spmm_forward_cuda<3, 4>(input, weight, rowPtr_bcsr, colIdx, values);
    case 4:
      return bcsr_spmm_forward_cuda<4, 4>(input, weight, rowPtr_bcsr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else if (bcsr_blkWarpNum == 8)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_spmm_forward_cuda<1, 8>(input, weight, rowPtr_bcsr, colIdx, values);
    case 2:
      return bcsr_spmm_forward_cuda<2, 8>(input, weight, rowPtr_bcsr, colIdx, values);
    case 3:
      return bcsr_spmm_forward_cuda<3, 8>(input, weight, rowPtr_bcsr, colIdx, values);
    case 4:
      return bcsr_spmm_forward_cuda<4, 8>(input, weight, rowPtr_bcsr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

std::vector<torch::Tensor> bcsr_spmm_backward(
    torch::Tensor g_input,
    torch::Tensor H,
    torch::Tensor weight,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(g_input);
  CHECK_INPUT(H);
  CHECK_INPUT(weight);
  CHECK_INPUT(rowPtr_bcsr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (bcsr_blkWarpNum == 4)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_spmm_backward_cuda<1, 4>(g_input, H, weight, rowPtr_bcsr, colIdx, values);
    case 2:
      return bcsr_spmm_backward_cuda<2, 4>(g_input, H, weight, rowPtr_bcsr, colIdx, values);
    case 3:
      return bcsr_spmm_backward_cuda<3, 4>(g_input, H, weight, rowPtr_bcsr, colIdx, values);
    case 4:
      return bcsr_spmm_backward_cuda<4, 4>(g_input, H, weight, rowPtr_bcsr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else if (bcsr_blkWarpNum == 8)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_spmm_backward_cuda<1, 8>(g_input, H, weight, rowPtr_bcsr, colIdx, values);
    case 2:
      return bcsr_spmm_backward_cuda<2, 8>(g_input, H, weight, rowPtr_bcsr, colIdx, values);
    case 3:
      return bcsr_spmm_backward_cuda<3, 8>(g_input, H, weight, rowPtr_bcsr, colIdx, values);
    case 4:
      return bcsr_spmm_backward_cuda<4, 8>(g_input, H, weight, rowPtr_bcsr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

torch::Tensor bcsr_split_spmm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(rowPtr_bcsr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (bcsr_split_blkWarpNum == 4)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_spmm_forward_cuda<1, 4>(input, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 2:
      return bcsr_split_spmm_forward_cuda<2, 4>(input, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 3:
      return bcsr_split_spmm_forward_cuda<3, 4>(input, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 4:
      return bcsr_split_spmm_forward_cuda<4, 4>(input, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else if (bcsr_split_blkWarpNum == 8)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_spmm_forward_cuda<1, 8>(input, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 2:
      return bcsr_split_spmm_forward_cuda<2, 8>(input, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 3:
      return bcsr_split_spmm_forward_cuda<3, 8>(input, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 4:
      return bcsr_split_spmm_forward_cuda<4, 8>(input, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

std::vector<torch::Tensor> bcsr_split_spmm_backward(
    torch::Tensor g_input,
    torch::Tensor H,
    torch::Tensor weight,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(g_input);
  CHECK_INPUT(H);
  CHECK_INPUT(weight);
  CHECK_INPUT(rowPtr_bcsr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (bcsr_split_blkWarpNum == 4)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_spmm_backward_cuda<1, 4>(g_input, H, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 2:
      return bcsr_split_spmm_backward_cuda<2, 4>(g_input, H, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 3:
      return bcsr_split_spmm_backward_cuda<3, 4>(g_input, H, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 4:
      return bcsr_split_spmm_backward_cuda<4, 4>(g_input, H, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else if (bcsr_split_blkWarpNum == 8)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_spmm_backward_cuda<1, 8>(g_input, H, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 2:
      return bcsr_split_spmm_backward_cuda<2, 8>(g_input, H, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 3:
      return bcsr_split_spmm_backward_cuda<3, 8>(g_input, H, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    case 4:
      return bcsr_split_spmm_backward_cuda<4, 8>(g_input, H, weight, rowPtr_bcsr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("split_csr", &split_csr, "split csr/bcsr with certain gran(CUDA)");
  m.def("split_csr_par", &split_csr_par, "split csr/bcsr with certain gran(CUDA) parallel");

  m.def("csr_spmm_forward", &csr_spmm_forward, "csr spmm (CUDA)");
  m.def("csr_spmm_backward", &csr_spmm_backward, "csr spmm (CUDA)");

  m.def("csr_split_spmm_forward", &csr_split_spmm_forward, "csr split spmm (CUDA)");
  m.def("csr_split_spmm_backward", &csr_split_spmm_backward, "csr split spmm (CUDA)");

  m.def("bcsr_spmm_forward", &bcsr_spmm_forward, "bcsr spmm (CUDA)");
  m.def("bcsr_spmm_backward", &bcsr_spmm_backward, "bcsr spmm (CUDA)");

  m.def("bcsr_split_spmm_forward", &bcsr_split_spmm_forward, "bcsr split spmm (CUDA)");
  m.def("bcsr_split_spmm_backward", &bcsr_split_spmm_backward, "bcsr split spmm (CUDA)");
}