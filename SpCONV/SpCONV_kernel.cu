#include "SpCONV.h"

#define WARP_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
__global__ void warmup() {}

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void csr_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    const int nr, const int dim);

template <typename scalar_t>
__global__ void csr_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    const int nr, const int dim, const int workerPerBlock);

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void csr_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_rows, const int dim);

template <typename scalar_t>
__global__ void csr_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_rows, const int dim, const int workerPerBlock);

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void bcsr_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    const int nblk_row, const int dim);

template <typename scalar_t>
__global__ void bcsr_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    const int nblk_row, const int dim, const int workerPerBlock);

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void bcsr_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim);

template <typename scalar_t>
__global__ void bcsr_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim, const int workerPerBlock);

template <int CF, int blkWarpNum>
void csr_spmm_cuda(
    torch::Tensor &output,
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    const int num_nodes = rowPtr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = output.size(1);           // embedding size
    int dim_x, dim_y;

    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(num_nodes, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        AT_DISPATCH_FLOATING_TYPES(input.type(), "csr-SpMM", ([&]
                                                              { csr_spmm_cuda_kernel<scalar_t><<<gridDim, blockDim>>>(
                                                                    output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                    input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                    rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                    colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                    values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                    num_nodes,
                                                                    dim,
                                                                    workerPerBlock); }));
    }
    else
    {
        dim_x = CEIL_DIV(num_nodes, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        if (is_residue)
            AT_DISPATCH_FLOATING_TYPES(input.type(), "csr-SpMM", ([&]
                                                                  { csr_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim>>>(
                                                                        output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                        input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                        rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                        colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                        values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                        num_nodes,
                                                                        dim); }));
        else
            AT_DISPATCH_FLOATING_TYPES(input.type(), "csr-SpMM", ([&]
                                                                  { csr_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim>>>(
                                                                        output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                        input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                        rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                        colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                        values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                        num_nodes,
                                                                        dim); }));
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <int CF, int blkWarpNum>
void csr_split_spmm_cuda(
    torch::Tensor &output,
    torch::Tensor input,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    const int split_rows = rowPtr_split.size(0) - 1; // number of splitted rows
    const int dim = output.size(1);                  // embedding size
    assert(target_row.size(0) == split_rows);
    int dim_x, dim_y;

    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(split_rows, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                        { csr_split_spmm_cuda_kernel<scalar_t><<<gridDim, blockDim>>>(
                                                                              output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              rowPtr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                              target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              split_rows,
                                                                              dim,
                                                                              workerPerBlock); }));
    }
    else
    {
        dim_x = CEIL_DIV(split_rows, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * 2;
        if (is_residue)
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { csr_split_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                                  output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  rowPtr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                  target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  split_rows,
                                                                                  dim); }));
        else
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { csr_split_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                                  output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  rowPtr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                  target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  split_rows,
                                                                                  dim); }));
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <int CF, int blkWarpNum>
void bcsr_spmm_cuda(
    torch::Tensor &output,
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    const int nblk_row = rowPtr_bcsr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = output.size(1);               // embedding size
    int dim_x, dim_y;

    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(nblk_row, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                        { bcsr_spmm_cuda_kernel<scalar_t><<<gridDim, blockDim>>>(
                                                                              output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              nblk_row,
                                                                              dim,
                                                                              workerPerBlock); }));
    }
    else
    {
        dim_x = CEIL_DIV(nblk_row, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        if (is_residue)
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { bcsr_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim>>>(
                                                                                  output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  nblk_row,
                                                                                  dim); }));
        else
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { bcsr_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim>>>(
                                                                                  output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  nblk_row,
                                                                                  dim); }));
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <int CF, int blkWarpNum>
void bcsr_split_spmm_cuda(
    torch::Tensor &output,
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    const int split_nblk_row = rowPtr_bcsr_split.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = output.size(1);                           // embedding size
    int dim_x, dim_y;

    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(split_nblk_row, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                        { bcsr_split_spmm_cuda_kernel<scalar_t><<<gridDim, blockDim>>>(
                                                                              output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              split_nblk_row,
                                                                              dim,
                                                                              workerPerBlock); }));
    }
    else
    {
        dim_x = CEIL_DIV(split_nblk_row, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;

        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * 3;

        if (is_residue)
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { bcsr_split_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                                  output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  split_nblk_row,
                                                                                  dim); }));
        else
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { bcsr_split_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                                  output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  split_nblk_row,
                                                                                  dim); }));
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void csr_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    const int nr, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    if (rowIdx >= nr)
        return;

    scalar_t val;
    scalar_t res[CF] = {0};
    int BrowIdx, offset = blockIdx.y * 32 * CF;
    // handle when dim is not multiple of 32

    int lb = __ldg(&rowPtr[rowIdx]);
    int hb = __ldg(&rowPtr[rowIdx + 1]);
    if (lb == hb)
        return;
    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId - offset, WARP_SIZE);
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
            val = values[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                if (i < CFnum)
                    res[i] += val * input[BrowIdx][offset + i * WARP_SIZE + laneId];
        }
#pragma unroll
        for (int i = 0; i < CF; i++)
            if (i < CFnum)
                output[rowIdx][offset + i * WARP_SIZE + laneId] = res[i];
    }
    else
    {
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
            val = values[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                res[i] += val * input[BrowIdx][offset + i * WARP_SIZE + laneId];
        }
#pragma unroll
        for (int i = 0; i < CF; i++)
            output[rowIdx][offset + i * WARP_SIZE + laneId] = res[i];
    }
}

template <typename scalar_t>
__global__ void csr_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    const int nr, const int dim, const int workerPerBlock)
{
    int workerId = threadIdx.x / dim;                    // workerId in a threadBlock
    int laneId = threadIdx.x % dim;                      // laneId in a worker
    int rowIdx = blockIdx.x * workerPerBlock + workerId; // workerPerBlock=blockDim.x/dim=WarpSize*blkWarpNum/dim
    if (rowIdx >= nr)
        return;

    scalar_t val;
    scalar_t res = 0;

    int lb = __ldg(&rowPtr[rowIdx]);
    int hb = __ldg(&rowPtr[rowIdx + 1]);
    if (lb == hb)
        return;
    for (; lb < hb; lb++)
    {
        val = values[lb];
        res += val * input[colIdx[lb]][laneId];
    }
    output[rowIdx][laneId] = res;
}

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void csr_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_rows, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;

    extern __shared__ int shared_mem[];
    int *col_sh = &shared_mem[(warpId << 5)];
    scalar_t *val_sh = (scalar_t *)(col_sh + blockDim.x);
    if (rowIdx >= split_rows)
        return;

    int BrowIdx, offset = blockIdx.y * 32 * CF;

    scalar_t val;
    scalar_t res[CF] = {0};
    int lb = __ldg(&rowPtr_split[rowIdx]);
    int hb = __ldg(&rowPtr_split[rowIdx + 1]);
    if (lb == hb)
        return;

    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId - offset, WARP_SIZE);
        for (int i = lb; i < hb; i += 32)
        {
            int g_idx = i + laneId;
            if (g_idx < hb)
            {
                col_sh[laneId] = colIdx[g_idx];
                val_sh[laneId] = values[g_idx];
            }
            __syncwarp();
            for (int j = 0; j < 32 && lb < hb; lb++, j++)
            {
                BrowIdx = col_sh[j];
                val = val_sh[j];
#pragma unroll
                for (int k = 0; k < CF; k++)
                    if (k < CFnum)
                        res[k] += input[BrowIdx][offset + k * WARP_SIZE + laneId] * val;
            }
        }
        rowIdx = __ldg(&target_row[rowIdx]);
#pragma unroll
        for (int i = 0; i < CF; i++)
            if (i < CFnum)
                atomicAdd(&output[rowIdx][offset + i * WARP_SIZE + laneId], res[i]);
    }
    else
    {
        for (int i = lb; i < hb; i += 32)
        {
            int g_idx = i + laneId;
            if (g_idx < hb)
            {
                col_sh[laneId] = colIdx[g_idx];
                val_sh[laneId] = values[g_idx];
            }
            __syncwarp();
            for (int j = 0; j < 32 && lb < hb; lb++, j++)
            {
                BrowIdx = col_sh[j];
                val = val_sh[j];
#pragma unroll
                for (int k = 0; k < CF; k++)
                    res[k] += input[BrowIdx][offset + k * WARP_SIZE + laneId] * val;
            }
        }
        rowIdx = __ldg(&target_row[rowIdx]);
#pragma unroll
        for (int i = 0; i < CF; i++)
            atomicAdd(&output[rowIdx][offset + i * WARP_SIZE + laneId], res[i]);
    }
}

template <typename scalar_t>
__global__ void csr_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_rows, const int dim, const int workerPerBlock)
{
    int workerId = threadIdx.x / dim;                    // workerId in a threadBlock
    int laneId = threadIdx.x % dim;                      // laneId in a worker
    int rowIdx = blockIdx.x * workerPerBlock + workerId; // workerPerBlock=blockDim.x/dim=WarpSize*blkWarpNum/dim
    if (rowIdx >= split_rows)
        return;

    scalar_t val;
    scalar_t res = 0;
    int BrowIdx;
    int lb = __ldg(&rowPtr_split[rowIdx]);
    int hb = __ldg(&rowPtr_split[rowIdx + 1]);
    if (lb == hb)
        return;
    for (; lb < hb; lb++)
    {
        BrowIdx = colIdx[lb];
        val = values[lb];
        res += val * input[BrowIdx][laneId];
    }
    rowIdx = __ldg(&target_row[rowIdx]);
    atomicAdd(&output[rowIdx][laneId], res);
}

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void bcsr_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    const int nblk_row, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    if (rowIdx >= nblk_row)
        return;

    scalar_t val[2];
    scalar_t res[2][CF] = {0};
    scalar_t Bval[CF] = {0};
    int BrowIdx, offset = blockIdx.y * 32 * CF;

    int lb = __ldg(&rowPtr_bcsr[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr[rowIdx + 1]);
    if (lb == hb)
        return;
    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId - offset, WARP_SIZE);
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
            val[0] = values[lb][0];
            val[1] = values[lb][1];
#pragma unroll
            for (int i = 0; i < CF; i++)
                if (i < CFnum)
                    Bval[i] = input[BrowIdx][offset + i * WARP_SIZE + laneId];

            if (val[0] != 0)
            {
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[0][i] += val[0] * Bval[i];
            }
            if (val[1] != 0)
            {
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[1][i] += val[1] * Bval[i];
            }
        }

#pragma unroll
        for (int i = 0; i < CF; i++)
        {
            if (i < CFnum)
            {
                output[rowIdx * 2][offset + i * WARP_SIZE + laneId] = res[0][i];
                output[rowIdx * 2 + 1][offset + i * WARP_SIZE + laneId] = res[1][i];
            }
        }
    }
    else
    {
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
            val[0] = values[lb][0];
            val[1] = values[lb][1];
#pragma unroll
            for (int i = 0; i < CF; i++)
                Bval[i] = input[BrowIdx][offset + i * WARP_SIZE + laneId];

            if (val[0] != 0)
            {
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[0][i] += val[0] * Bval[i];
            }
            if (val[1] != 0)
            {
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[1][i] += val[1] * Bval[i];
            }
        }
#pragma unroll
        for (int i = 0; i < CF; i++)
        {
            output[rowIdx * 2][offset + i * WARP_SIZE + laneId] = res[0][i];
            output[rowIdx * 2 + 1][offset + i * WARP_SIZE + laneId] = res[1][i];
        }
    }
}

template <typename scalar_t>
__global__ void bcsr_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    const int nblk_row, const int dim, const int workerPerBlock)
{
    int workerId = threadIdx.x / dim;
    int laneId = threadIdx.x % dim;
    int rowIdx = blockIdx.x * workerPerBlock + workerId;
    if (rowIdx >= nblk_row)
        return;

    scalar_t val[2];
    scalar_t res[2] = {0};
    scalar_t Bval;

    int lb = __ldg(&rowPtr_bcsr[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr[rowIdx + 1]);
    if (lb == hb)
        return;
    for (; lb < hb; lb++)
    {
        val[0] = values[lb][0];
        val[1] = values[lb][1];
        Bval = input[colIdx[lb]][laneId];

        if (val[0] != 0)
            res[0] += val[0] * Bval;

        if (val[1] != 0)
            res[1] += val[1] * Bval;
    }
    output[rowIdx * 2][laneId] = res[0];
    output[rowIdx * 2 + 1][laneId] = res[1];
}

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void bcsr_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    if (rowIdx >= split_nblk_row)
        return;

    extern __shared__ int shared_mem[];

    int *col_sh = &shared_mem[(warpId * WARP_SIZE)];
    float2 *val_sh = (float2 *)&shared_mem[(warpId << 6) + blockDim.x];

    float2 val;
    scalar_t res[2][CF] = {0};
    scalar_t Bval[CF];
    int BrowIdx, offset = blockIdx.y * 32 * CF;
    int lb = __ldg(&rowPtr_bcsr_split[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr_split[rowIdx + 1]);
    if (lb == hb)
        return;

    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId - offset, WARP_SIZE);
        for (int i = lb; i < hb; i += 32)
        {
            int g_idx = i + laneId;
            if (g_idx < hb)
            {
                col_sh[laneId] = colIdx[g_idx];
                val_sh[laneId] = *(float2 *)(&values[g_idx][0]);
            }
            __syncwarp();
            for (int j = 0; j < 32 && lb < hb; lb++, j++)
            {
                BrowIdx = col_sh[j];
                val = val_sh[j];
#pragma unroll
                for (int k = 0; k < CF; k++)
                    if (k < CFnum)
                        Bval[k] = input[BrowIdx][offset + k * WARP_SIZE + laneId];

                if (val.x != 0)
                {
#pragma unroll
                    for (int k = 0; k < CF; k++)
                        res[0][k] += val.x * Bval[k];
                }
                if (val.y != 0)
                {
#pragma unroll
                    for (int k = 0; k < CF; k++)
                        res[1][k] += val.y * Bval[k];
                }
            }
        }
        rowIdx = __ldg(&target_row[rowIdx]) * 2;
#pragma unroll
        for (int i = 0; i < CF; i++)
        {
            if (i < CFnum)
            {
                atomicAdd(&output[rowIdx][offset + i * WARP_SIZE + laneId], res[0][i]);
                atomicAdd(&output[rowIdx + 1][offset + i * WARP_SIZE + laneId], res[1][i]);
            }
        }
    }
    else
    {
        for (int i = lb; i < hb; i += 32)
        {
            int g_idx = i + laneId;
            if (g_idx < hb)
            {
                col_sh[laneId] = colIdx[g_idx];
                val_sh[laneId] = *(float2 *)(&values[g_idx][0]);
            }
            __syncwarp();
            for (int j = 0; j < 32 && lb < hb; lb++, j++)
            {
                BrowIdx = col_sh[j];
                val = val_sh[j];
#pragma unroll
                for (int k = 0; k < CF; k++)
                    Bval[k] = input[BrowIdx][offset + k * WARP_SIZE + laneId];

                if (val.x != 0)
                {
#pragma unroll
                    for (int k = 0; k < CF; k++)
                        res[0][k] += val.x * Bval[k];
                }
                if (val.y != 0)
                {
#pragma unroll
                    for (int k = 0; k < CF; k++)
                        res[1][k] += val.y * Bval[k];
                }
            }
        }
        rowIdx = __ldg(&target_row[rowIdx]) * 2;
#pragma unroll
        for (int i = 0; i < CF; i++)
        {

            atomicAdd(&output[rowIdx][offset + i * WARP_SIZE + laneId], res[0][i]);
            atomicAdd(&output[rowIdx + 1][offset + i * WARP_SIZE + laneId], res[1][i]);
        }
    }
}

template <typename scalar_t>
__global__ void bcsr_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim, const int workerPerBlock)
{
    int workerId = threadIdx.x / dim;
    int laneId = threadIdx.x % dim;
    int rowIdx = blockIdx.x * workerPerBlock + workerId;
    if (rowIdx >= split_nblk_row)
        return;
    scalar_t val[2];
    scalar_t res[2] = {0};
    scalar_t Bval;

    int lb = __ldg(&rowPtr_bcsr_split[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr_split[rowIdx + 1]);
    if (lb == hb)
        return;
    for (; lb < hb; lb++)
    {
        val[0] = values[lb][0];
        val[1] = values[lb][1];
        Bval = input[colIdx[lb]][laneId];
        if (val[0] != 0)
            res[0] += val[0] * Bval;
        if (val[1] != 0)
            res[1] += val[1] * Bval;
    }
    rowIdx = __ldg(&target_row[rowIdx]) * 2;
    atomicAdd(&output[rowIdx][laneId], res[0]);
    atomicAdd(&output[rowIdx + 1][laneId], res[1]);
}

template void csr_spmm_cuda<1, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template void csr_spmm_cuda<2, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template void csr_spmm_cuda<3, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template void csr_spmm_cuda<4, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template void csr_spmm_cuda<1, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template void csr_spmm_cuda<2, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template void csr_spmm_cuda<3, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template void csr_spmm_cuda<4, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);

template void csr_split_spmm_cuda<1, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void csr_split_spmm_cuda<2, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void csr_split_spmm_cuda<3, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void csr_split_spmm_cuda<4, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void csr_split_spmm_cuda<1, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void csr_split_spmm_cuda<2, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void csr_split_spmm_cuda<3, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void csr_split_spmm_cuda<4, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template void bcsr_spmm_cuda<1, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template void bcsr_spmm_cuda<2, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template void bcsr_spmm_cuda<3, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template void bcsr_spmm_cuda<4, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template void bcsr_spmm_cuda<1, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template void bcsr_spmm_cuda<2, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template void bcsr_spmm_cuda<3, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template void bcsr_spmm_cuda<4, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);

template void bcsr_split_spmm_cuda<1, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void bcsr_split_spmm_cuda<2, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void bcsr_split_spmm_cuda<3, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void bcsr_split_spmm_cuda<4, 4>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void bcsr_split_spmm_cuda<1, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void bcsr_split_spmm_cuda<2, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void bcsr_split_spmm_cuda<3, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template void bcsr_split_spmm_cuda<4, 8>(torch::Tensor &output, torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
