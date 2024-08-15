#include <torch/extension.h>
#include <vector>
#include "sddmm.h"

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)

torch::Tensor csr_sddmm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int csr_CF,
    int csr_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(rowPtr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (csr_blkWarpNum == 4)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_sddmm_cuda<1, 4>(A, B, rowPtr, colIdx, values);
    case 2:
      return csr_sddmm_cuda<2, 4>(A, B, rowPtr, colIdx, values);
    case 3:
      return csr_sddmm_cuda<3, 4>(A, B, rowPtr, colIdx, values);
    case 4:
      return csr_sddmm_cuda<4, 4>(A, B, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (csr_blkWarpNum == 8)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_sddmm_cuda<1, 8>(A, B, rowPtr, colIdx, values);
    case 2:
      return csr_sddmm_cuda<2, 8>(A, B, rowPtr, colIdx, values);
    case 3:
      return csr_sddmm_cuda<3, 8>(A, B, rowPtr, colIdx, values);
    case 4:
      return csr_sddmm_cuda<4, 8>(A, B, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
  {
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
}

torch::Tensor csr_split_sddmm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int csr_split_CF,
    int csr_split_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(rowPtr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (csr_split_blkWarpNum == 4)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_sddmm_cuda<1, 4>(A, B, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_sddmm_cuda<2, 4>(A, B, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_sddmm_cuda<3, 4>(A, B, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_sddmm_cuda<4, 4>(A, B, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (csr_split_blkWarpNum == 8)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_sddmm_cuda<1, 8>(A, B, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_sddmm_cuda<2, 8>(A, B, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_sddmm_cuda<3, 8>(A, B, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_sddmm_cuda<4, 8>(A, B, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

torch::Tensor bcsr_sddmm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(rowPtr_bcsr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (bcsr_blkWarpNum == 4)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_sddmm_cuda<1, 4>(A, B, rowPtr_bcsr, colIdx, values);
    case 2:
      return bcsr_sddmm_cuda<2, 4>(A, B, rowPtr_bcsr, colIdx, values);
    case 3:
      return bcsr_sddmm_cuda<3, 4>(A, B, rowPtr_bcsr, colIdx, values);
    case 4:
      return bcsr_sddmm_cuda<4, 4>(A, B, rowPtr_bcsr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (bcsr_blkWarpNum == 8)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_sddmm_cuda<1, 8>(A, B, rowPtr_bcsr, colIdx, values);
    case 2:
      return bcsr_sddmm_cuda<2, 8>(A, B, rowPtr_bcsr, colIdx, values);
    case 3:
      return bcsr_sddmm_cuda<3, 8>(A, B, rowPtr_bcsr, colIdx, values);
    case 4:
      return bcsr_sddmm_cuda<4, 8>(A, B, rowPtr_bcsr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

torch::Tensor bcsr_split_sddmm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(rowPtr_bcsr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (bcsr_split_blkWarpNum == 4)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_sddmm_cuda<1, 4>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
    case 2:
      return bcsr_split_sddmm_cuda<2, 4>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
    case 3:
      return bcsr_split_sddmm_cuda<3, 4>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
    case 4:
      return bcsr_split_sddmm_cuda<4, 4>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (bcsr_split_blkWarpNum == 8)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_sddmm_cuda<1, 8>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
    case 2:
      return bcsr_split_sddmm_cuda<2, 8>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
    case 3:
      return bcsr_split_sddmm_cuda<3, 8>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
    case 4:
      return bcsr_split_sddmm_cuda<4, 8>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

torch::Tensor vec_sddmm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
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
        return vec_sddmm_cuda<1, 4, 3>(A, B, rowPtr_bcsr, colIdx, values);
      case 2:
        return vec_sddmm_cuda<2, 4, 3>(A, B, rowPtr_bcsr, colIdx, values);
      case 3:
        return vec_sddmm_cuda<3, 4, 3>(A, B, rowPtr_bcsr, colIdx, values);
      case 4:
        return vec_sddmm_cuda<4, 4, 3>(A, B, rowPtr_bcsr, colIdx, values);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else if (bcsr_blkWarpNum == 8)
    {
      switch (bcsr_CF)
      {
      case 1:
        return vec_sddmm_cuda<1, 8, 3>(A, B, rowPtr_bcsr, colIdx, values);
      case 2:
        return vec_sddmm_cuda<2, 8, 3>(A, B, rowPtr_bcsr, colIdx, values);
      case 3:
        return vec_sddmm_cuda<3, 8, 3>(A, B, rowPtr_bcsr, colIdx, values);
      case 4:
        return vec_sddmm_cuda<4, 8, 3>(A, B, rowPtr_bcsr, colIdx, values);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else
      TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
  else
    TORCH_CHECK(false, "vec_size must be 3");

  return values;
}

torch::Tensor vec_split_sddmm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
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
        return vec_split_sddmm_cuda<1, 4, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
      case 2:
        return vec_split_sddmm_cuda<2, 4, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
      case 3:
        return vec_split_sddmm_cuda<3, 4, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
      case 4:
        return vec_split_sddmm_cuda<4, 4, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else if (bcsr_split_blkWarpNum == 8)
    {
      switch (bcsr_split_CF)
      {
      case 1:
        return vec_split_sddmm_cuda<1, 8, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
      case 2:
        return vec_split_sddmm_cuda<2, 8, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
      case 3:
        return vec_split_sddmm_cuda<3, 8, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
      case 4:
        return vec_split_sddmm_cuda<4, 8, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row);
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

int csr_sddmm_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int csr_CF,
    int csr_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(rowPtr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (csr_blkWarpNum == 4)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_sddmm_cuda_profile<1, 4>(A, B, rowPtr, colIdx, values);
    case 2:
      return csr_sddmm_cuda_profile<2, 4>(A, B, rowPtr, colIdx, values);
    case 3:
      return csr_sddmm_cuda_profile<3, 4>(A, B, rowPtr, colIdx, values);
    case 4:
      return csr_sddmm_cuda_profile<4, 4>(A, B, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (csr_blkWarpNum == 8)
  {
    switch (csr_CF)
    {
    case 1:
      return csr_sddmm_cuda_profile<1, 8>(A, B, rowPtr, colIdx, values);
    case 2:
      return csr_sddmm_cuda_profile<2, 8>(A, B, rowPtr, colIdx, values);
    case 3:
      return csr_sddmm_cuda_profile<3, 8>(A, B, rowPtr, colIdx, values);
    case 4:
      return csr_sddmm_cuda_profile<4, 8>(A, B, rowPtr, colIdx, values);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
  {
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
  }
}

int csr_split_sddmm_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int csr_split_CF,
    int csr_split_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(rowPtr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (csr_split_blkWarpNum == 4)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_sddmm_cuda_profile<1, 4>(A, B, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_sddmm_cuda_profile<2, 4>(A, B, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_sddmm_cuda_profile<3, 4>(A, B, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_sddmm_cuda_profile<4, 4>(A, B, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (csr_split_blkWarpNum == 8)
  {
    switch (csr_split_CF)
    {
    case 1:
      return csr_split_sddmm_cuda_profile<1, 8>(A, B, rowPtr_split, colIdx, values, target_row);
    case 2:
      return csr_split_sddmm_cuda_profile<2, 8>(A, B, rowPtr_split, colIdx, values, target_row);
    case 3:
      return csr_split_sddmm_cuda_profile<3, 8>(A, B, rowPtr_split, colIdx, values, target_row);
    case 4:
      return csr_split_sddmm_cuda_profile<4, 8>(A, B, rowPtr_split, colIdx, values, target_row);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}
// special case when vec_size=2 float2 load could be used in kernel
int bcsr_sddmm_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(rowPtr_bcsr);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  if (bcsr_blkWarpNum == 4)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_sddmm_cuda_profile<1, 4>(A, B, rowPtr_bcsr, colIdx, values, nnz);
    case 2:
      return bcsr_sddmm_cuda_profile<2, 4>(A, B, rowPtr_bcsr, colIdx, values, nnz);
    case 3:
      return bcsr_sddmm_cuda_profile<3, 4>(A, B, rowPtr_bcsr, colIdx, values, nnz);
    case 4:
      return bcsr_sddmm_cuda_profile<4, 4>(A, B, rowPtr_bcsr, colIdx, values, nnz);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (bcsr_blkWarpNum == 8)
  {
    switch (bcsr_CF)
    {
    case 1:
      return bcsr_sddmm_cuda_profile<1, 8>(A, B, rowPtr_bcsr, colIdx, values, nnz);
    case 2:
      return bcsr_sddmm_cuda_profile<2, 8>(A, B, rowPtr_bcsr, colIdx, values, nnz);
    case 3:
      return bcsr_sddmm_cuda_profile<3, 8>(A, B, rowPtr_bcsr, colIdx, values, nnz);
    case 4:
      return bcsr_sddmm_cuda_profile<4, 8>(A, B, rowPtr_bcsr, colIdx, values, nnz);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}
// special case when vec_size=2 float2 load could be used in kernel
int bcsr_split_sddmm_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(rowPtr_bcsr_split);
  CHECK_INPUT(colIdx);
  CHECK_INPUT(values);
  CHECK_INPUT(target_row);
  if (bcsr_split_blkWarpNum == 4)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_sddmm_cuda_profile<1, 4>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 2:
      return bcsr_split_sddmm_cuda_profile<2, 4>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 3:
      return bcsr_split_sddmm_cuda_profile<3, 4>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 4:
      return bcsr_split_sddmm_cuda_profile<4, 4>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else if (bcsr_split_blkWarpNum == 8)
  {
    switch (bcsr_split_CF)
    {
    case 1:
      return bcsr_split_sddmm_cuda_profile<1, 8>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 2:
      return bcsr_split_sddmm_cuda_profile<2, 8>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 3:
      return bcsr_split_sddmm_cuda_profile<3, 8>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    case 4:
      return bcsr_split_sddmm_cuda_profile<4, 8>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
    default:
      TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
    }
  }
  else
    TORCH_CHECK(false, "blkWarpNum must be 4 or 8");
}

// specialized case of svb (vec_size=2)
int vec_sddmm_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz,
    int bcsr_CF,
    int bcsr_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
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
        return vec_sddmm_cuda_profile<1, 4, 3>(A, B, rowPtr_bcsr, colIdx, values, nnz);
      case 2:
        return vec_sddmm_cuda_profile<2, 4, 3>(A, B, rowPtr_bcsr, colIdx, values, nnz);
      case 3:
        return vec_sddmm_cuda_profile<3, 4, 3>(A, B, rowPtr_bcsr, colIdx, values, nnz);
      case 4:
        return vec_sddmm_cuda_profile<4, 4, 3>(A, B, rowPtr_bcsr, colIdx, values, nnz);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else if (bcsr_blkWarpNum == 8)
    {
      switch (bcsr_CF)
      {
      case 1:
        return vec_sddmm_cuda_profile<1, 8, 3>(A, B, rowPtr_bcsr, colIdx, values, nnz);
      case 2:
        return vec_sddmm_cuda_profile<2, 8, 3>(A, B, rowPtr_bcsr, colIdx, values, nnz);
      case 3:
        return vec_sddmm_cuda_profile<3, 8, 3>(A, B, rowPtr_bcsr, colIdx, values, nnz);
      case 4:
        return vec_sddmm_cuda_profile<4, 8, 3>(A, B, rowPtr_bcsr, colIdx, values, nnz);
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
int vec_split_sddmm_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz,
    int bcsr_split_CF,
    int bcsr_split_blkWarpNum)
{
  CHECK_INPUT(A);
  CHECK_INPUT(B);
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
        return vec_split_sddmm_cuda_profile<1, 4, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 2:
        return vec_split_sddmm_cuda_profile<2, 4, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 3:
        return vec_split_sddmm_cuda_profile<3, 4, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 4:
        return vec_split_sddmm_cuda_profile<4, 4, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      default:
        TORCH_CHECK(false, "CF must be 1, 2, 3, 4");
      }
    }
    else if (bcsr_split_blkWarpNum == 8)
    {
      switch (bcsr_split_CF)
      {
      case 1:
        return vec_split_sddmm_cuda_profile<1, 8, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 2:
        return vec_split_sddmm_cuda_profile<2, 8, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 3:
        return vec_split_sddmm_cuda_profile<3, 8, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
      case 4:
        return vec_split_sddmm_cuda_profile<4, 8, 3>(A, B, rowPtr_bcsr_split, colIdx, values, target_row, nnz);
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

torch::Tensor genmask(
    torch::Tensor rowPtr_bcsr,
    torch::Tensor rowPtr,
    torch::Tensor blkval,
    int nnz)
{
  CHECK_INPUT(blkval);
  CHECK_INPUT(rowPtr_bcsr);
  // val is 1d tensor
  return genmask_cuda(rowPtr_bcsr, rowPtr, blkval, nnz);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("csr_sddmm", &csr_sddmm, "csr sddmm (CUDA)");
  m.def("csr_split_sddmm", &csr_split_sddmm, "csr split sddmm (CUDA)");

  m.def("bcsr_sddmm", &bcsr_sddmm, "bcsr sddmm (CUDA)");
  m.def("bcsr_split_sddmm", &bcsr_split_sddmm, "bcsr split sddmm (CUDA)");
  // vec_size>1
  m.def("vec_sddmm", &vec_sddmm, "bcsr sddmm (CUDA)");
  m.def("vec_split_sddmm", &vec_split_sddmm, "bcsr split sddmm (CUDA)");

  m.def("csr_sddmm_profile", &csr_sddmm_profile, "csr sddmm profiling inside CPP ");
  m.def("csr_split_sddmm_profile", &csr_split_sddmm_profile, "csr split sddmm profiling inside CPP ");
  // special case when vec_size=2 float2 load could be used in kernel
  m.def("bcsr_sddmm_profile", &bcsr_sddmm_profile, "bcsr sddmm profiling inside CPP ");
  m.def("bcsr_split_sddmm_profile", &bcsr_split_sddmm_profile, "bcsr split sddmm profiling inside CPP ");
  // vec_size>1
  m.def("vec_sddmm_profile", &vec_sddmm_profile, "vec sddmm profiling inside CPP ");
  m.def("vec_split_sddmm_profile", &vec_split_sddmm_profile, "vec split sddmm profiling inside CPP ");

  // vendor-provided SDDMM by NVIDIA
  m.def("cusparse_sddmm_compute", &cusparse_sddmm_compute, "cusparse compute (CUDA)");
  m.def("cusparse_sddmm_profile", &cusparse_sddmm_profile, "cusparse compute (CUDA)");

  // 2d val to 1d val
  m.def("genmask", &genmask, "dval to val (CUDA)");
}