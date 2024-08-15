#include "SpMM.h"

#define WARP_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define PROFILE 200

__global__ void warmup()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

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

template <typename scalar_t, int CF, int blkWarpNum, int vec_size, bool is_residue>
__global__ void vec_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    const int nblk_row, const int dim);

template <typename scalar_t, int vec_size>
__global__ void vec_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    const int nblk_row, const int dim, const int workerPerBlock);

template <typename scalar_t, int CF, int blkWarpNum, int vec_size, bool is_residue>
__global__ void vec_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim);

template <typename scalar_t, int vec_size>
__global__ void vec_split_spmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim, const int workerPerBlock);

template <int CF, int blkWarpNum>
torch::Tensor csr_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
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

    return output;
}

template <int CF, int blkWarpNum>
torch::Tensor csr_split_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    auto output = torch::zeros_like(input);
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

    return output;
}

template <int CF, int blkWarpNum>
torch::Tensor bcsr_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    const int padded = (input.size(0) + 2 - 1) / 2 * 2;
    auto output = torch::zeros({padded, input.size(1)}, input.options());
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

    return output;
}

template <int CF, int blkWarpNum>
torch::Tensor bcsr_split_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    const int padded = (input.size(0) + 2 - 1) / 2 * 2;
    auto output = torch::zeros({padded, input.size(1)}, input.options());
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
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * 3;
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
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

    return output;
}

template <int CF, int blkWarpNum, int vec_size>
torch::Tensor vec_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    // auto output = torch::zeros_like(input);
    const int padded = (input.size(0) + vec_size - 1) / vec_size * vec_size;
    auto output = torch::zeros({padded, input.size(1)}, input.options());
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
                                                                        { vec_spmm_cuda_kernel<scalar_t, vec_size><<<gridDim, blockDim>>>(
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
                                                                            { vec_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, true><<<gridDim, blockDim>>>(
                                                                                  output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  nblk_row,
                                                                                  dim); }));
        else
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { vec_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, false><<<gridDim, blockDim>>>(
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

    return output;
}

template <int CF, int blkWarpNum, int vec_size>
torch::Tensor vec_split_spmm_cuda(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    const int padded = (input.size(0) + vec_size - 1) / vec_size * vec_size;
    auto output = torch::zeros({padded, input.size(1)}, input.options());
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
                                                                        { vec_split_spmm_cuda_kernel<scalar_t, vec_size><<<gridDim, blockDim>>>(
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
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * (vec_size + 1);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        if (is_residue)
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { vec_split_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, true><<<gridDim, blockDim, smem>>>(
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
                                                                            { vec_split_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, false><<<gridDim, blockDim, smem>>>(
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

    return output;
}

template <int CF, int blkWarpNum>
int csr_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int num_nodes = rowPtr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = output.size(1);           // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(num_nodes, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        cudaEventRecord(start, 0);
        for (int i = 0; i < PROFILE; i++)
        {
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
    }
    else
    {
        dim_x = CEIL_DIV(num_nodes, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        cudaEventRecord(start, 0);
        if (is_residue)
            for (int i = 0; i < PROFILE; i++)
            {
                AT_DISPATCH_FLOATING_TYPES(input.type(), "csr-SpMM", ([&]
                                                                      { csr_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim>>>(
                                                                            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                            rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                            colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                            values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            num_nodes,
                                                                            dim); }));
            }
        else
            for (int i = 0; i < PROFILE; i++)
            {
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
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = colIdx.size(0) / 1e6 * dim * 2;
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return (int)(gflop / (milliseconds / PROFILE));
}

template <int CF, int blkWarpNum>
int csr_split_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    auto output = torch::zeros_like(input);
    const int split_rows = rowPtr_split.size(0) - 1; // number of splitted rows
    const int dim = output.size(1);                  // embedding size
    assert(target_row.size(0) == split_rows);
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(split_rows, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        cudaEventRecord(start, 0);
        for (int i = 0; i < PROFILE; i++)
        {
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
    }
    else
    {
        dim_x = CEIL_DIV(split_rows, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * 2;
        cudaEventRecord(start, 0);
        if (is_residue)
            for (int i = 0; i < PROFILE; i++)
            {
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
            }
        else
            for (int i = 0; i < PROFILE; i++)
            {
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
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = colIdx.size(0) / 1e6 * dim * 2;
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return (int)(gflop / (milliseconds / PROFILE));
}

template <int CF, int blkWarpNum>
int bcsr_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz)
{
    const int padded = (input.size(0) + 2 - 1) / 2 * 2;
    auto output = torch::zeros({padded, input.size(1)}, input.options());
    const int nblk_row = rowPtr_bcsr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = output.size(1);               // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }
    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(nblk_row, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        cudaEventRecord(start, 0);
        for (int i = 0; i < PROFILE; i++)
        {
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
    }
    else
    {
        dim_x = CEIL_DIV(nblk_row, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        cudaEventRecord(start, 0);
        if (is_residue)
            for (int i = 0; i < PROFILE; i++)
            {
                AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                                { bcsr_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim>>>(
                                                                                      output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      nblk_row,
                                                                                      dim); }));
            }
        else
            for (int i = 0; i < PROFILE; i++)
            {
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
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = nnz / 1e6 * dim * 2;
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return (int)(gflop / (milliseconds / PROFILE));
}

template <int CF, int blkWarpNum>
int bcsr_split_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz)
{
    const int padded = (input.size(0) + 2 - 1) / 2 * 2;
    auto output = torch::zeros({padded, input.size(1)}, input.options());
    const int split_nblk_row = rowPtr_bcsr_split.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = output.size(1);                           // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(split_nblk_row, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        cudaEventRecord(start, 0);
        for (int i = 0; i < PROFILE; i++)
        {
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
    }
    else
    {
        dim_x = CEIL_DIV(split_nblk_row, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * 3;
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        cudaEventRecord(start, 0);
        if (is_residue)
            for (int i = 0; i < PROFILE; i++)
            {
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
            }
        else
            for (int i = 0; i < PROFILE; i++)
            {
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
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = nnz / 1e6 * dim * 2;
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return (int)(gflop / (milliseconds / PROFILE));
}

template <int CF, int blkWarpNum, int vec_size>
int vec_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz)
{
    // auto output = torch::zeros_like(input);
    const int padded = (input.size(0) + vec_size - 1) / vec_size * vec_size;
    auto output = torch::zeros({padded, input.size(1)}, input.options());
    const int nblk_row = rowPtr_bcsr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = output.size(1);               // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }
    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(nblk_row, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        cudaEventRecord(start, 0);
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { vec_spmm_cuda_kernel<scalar_t, vec_size><<<gridDim, blockDim>>>(
                                                                                  output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                  values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                  nblk_row,
                                                                                  dim,
                                                                                  workerPerBlock); }));
        }
    }
    else
    {
        dim_x = CEIL_DIV(nblk_row, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        cudaEventRecord(start, 0);
        if (is_residue)
            for (int i = 0; i < PROFILE; i++)
            {
                AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                                { vec_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, true><<<gridDim, blockDim>>>(
                                                                                      output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      nblk_row,
                                                                                      dim); }));
            }
        else
            for (int i = 0; i < PROFILE; i++)
            {
                AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                                { vec_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, false><<<gridDim, blockDim>>>(
                                                                                      output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      nblk_row,
                                                                                      dim); }));
            }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = nnz / 1e6 * dim * 2;
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return (int)(gflop / (milliseconds / PROFILE));
}

template <int CF, int blkWarpNum, int vec_size>
int vec_split_spmm_cuda_profile(
    torch::Tensor input,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz)
{
    const int padded = (input.size(0) + vec_size - 1) / vec_size * vec_size;
    auto output = torch::zeros({padded, input.size(1)}, input.options());
    const int split_nblk_row = rowPtr_bcsr_split.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = output.size(1);                           // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }
    if (CF == 1 && (dim < 32 && (dim & (dim - 1)) == 0))
    {
        int workerPerBlock = blkWarpNum * WARP_SIZE / dim;
        dim_x = CEIL_DIV(split_nblk_row, workerPerBlock);
        dim_y = 1;
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        cudaEventRecord(start, 0);
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                            { vec_split_spmm_cuda_kernel<scalar_t, vec_size><<<gridDim, blockDim>>>(
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
    }
    else
    {
        dim_x = CEIL_DIV(split_nblk_row, blkWarpNum);
        dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
        dim3 gridDim(dim_x, dim_y, 1);
        dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
        const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * (1 + vec_size);
        const bool is_residue = dim % (CF * WARP_SIZE) != 0;
        cudaEventRecord(start, 0);
        if (is_residue)
            for (int i = 0; i < PROFILE; i++)
            {
                AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                                { vec_split_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, true><<<gridDim, blockDim, smem>>>(
                                                                                      output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      split_nblk_row,
                                                                                      dim); }));
            }

        else
            for (int i = 0; i < PROFILE; i++)
            {
                AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&]
                                                                                { vec_split_spmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, false><<<gridDim, blockDim, smem>>>(
                                                                                      output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                      target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                      split_nblk_row,
                                                                                      dim); }));
            }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = nnz / 1e6 * dim * 2;
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return (int)(gflop / (milliseconds / PROFILE));
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
// a thread warp is responsible for handling multiple rows when dim<32 and dim = 2^n
// CF = 1. one row one worker
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

template <typename scalar_t, int CF, int blkWarpNum, int vec_size, bool is_residue>
__global__ void vec_spmm_cuda_kernel(
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

    scalar_t val[vec_size] = {0};
    scalar_t res[vec_size][CF] = {0};
    scalar_t Bval[CF] = {0};
    int BrowIdx, offset = blockIdx.y * WARP_SIZE * CF;

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
#pragma unroll
            for (int i = 0; i < vec_size; i++)
                val[i] = values[lb][i];

#pragma unroll
            for (int i = 0; i < CF; i++)
                if (i < CFnum)
                    Bval[i] = input[BrowIdx][offset + i * WARP_SIZE + laneId];
#pragma unroll
            for (int i = 0; i < vec_size; i++)
            {
                if (val[i] != 0)
                {
#pragma unroll
                    for (int j = 0; j < CF; j++)
                        res[i][j] += val[i] * Bval[j];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < CF; i++)
        {
            if (i < CFnum)
            {
#pragma unroll
                for (int j = 0; j < vec_size; j++)
                    output[rowIdx * vec_size + j][offset + i * WARP_SIZE + laneId] = res[j][i];
            }
        }
    }
    else
    {
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < vec_size; i++)
                val[i] = values[lb][i];

#pragma unroll
            for (int i = 0; i < CF; i++)
                Bval[i] = input[BrowIdx][offset + i * WARP_SIZE + laneId];
#pragma unroll
            for (int i = 0; i < vec_size; i++)
            {
                if (val[i] != 0)
                {
#pragma unroll
                    for (int j = 0; j < CF; j++)
                        res[i][j] += val[i] * Bval[j];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < CF; i++)
        {
#pragma unroll
            for (int j = 0; j < vec_size; j++)
                output[rowIdx * vec_size + j][offset + i * WARP_SIZE + laneId] = res[j][i];
        }
    }
}

template <typename scalar_t, int vec_size>
__global__ void vec_spmm_cuda_kernel(
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

    scalar_t val[vec_size] = {0};
    scalar_t res[vec_size] = {0};
    scalar_t Bval;

    int lb = __ldg(&rowPtr_bcsr[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr[rowIdx + 1]);
    if (lb == hb)
        return;

    for (; lb < hb; lb++)
    {
#pragma unroll
        for (int i = 0; i < vec_size; i++)
            val[i] = values[lb][i];
        Bval = input[colIdx[lb]][laneId];

#pragma unroll
        for (int i = 0; i < vec_size; i++)
            if (val[i] != 0)
                res[i] += val[i] * Bval;
    }
#pragma unroll
    for (int i = 0; i < vec_size; i++)
        output[rowIdx * vec_size + i][laneId] = res[i];
}

template <typename scalar_t, int CF, int blkWarpNum, int vec_size, bool is_residue>
__global__ void vec_split_spmm_cuda_kernel(
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
    scalar_t *val_sh = (scalar_t *)&shared_mem[(warpId * WARP_SIZE * vec_size) + blockDim.x];

    scalar_t val[vec_size] = {0};
    scalar_t res[vec_size][CF] = {0};
    scalar_t Bval[CF];
    int BrowIdx, offset = blockIdx.y * WARP_SIZE * CF;
    int lb = __ldg(&rowPtr_bcsr_split[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr_split[rowIdx + 1]);
    if (lb == hb)
        return;
    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId - offset, WARP_SIZE);
        for (int i = lb; i < hb; i += WARP_SIZE)
        {
            int g_idx = i + laneId;
            if (g_idx < hb)
                col_sh[laneId] = colIdx[g_idx];
            // coaleced access
            scalar_t *val_arr = (scalar_t *)(&values[i][0]);
#pragma unroll
            for (int j = 0; j < vec_size; j++)
                if (j * WARP_SIZE + laneId < (hb - lb) * vec_size)
                    val_sh[j * WARP_SIZE + laneId] = val_arr[j * WARP_SIZE + laneId];
            // uncoaleced access
            //             if (g_idx < hb)
            //             {
            //                 col_sh[laneId] = colIdx[g_idx];
            //                 scalar_t *val_arr = (scalar_t *)(&values[i][0]);
            // #pragma unroll
            //                 for (int j = 0; j < vec_size; j++)
            //                     val_sh[j * WARP_SIZE + laneId] = val_arr[j * WARP_SIZE + laneId];
            //                     // val_sh[j * WARP_SIZE + laneId] = values[g_idx][j];
            //             }
            __syncwarp();
            for (int j = 0; j < WARP_SIZE && lb < hb; lb++, j++)
            {
                BrowIdx = col_sh[j];
#pragma unroll
                for (int k = 0; k < vec_size; k++)
                    val[k] = val_sh[k + j * vec_size];
                    // val[k] = val_sh[k * WARP_SIZE + j];
#pragma unroll
                for (int k = 0; k < CF; k++)
                    if (k < CFnum)
                        Bval[k] = input[BrowIdx][offset + k * WARP_SIZE + laneId];
#pragma unroll
                for (int k = 0; k < vec_size; k++)
                {
                    if (val[k] != 0)
                    {
#pragma unroll
                        for (int l = 0; l < CF; l++)
                            res[k][l] += val[k] * Bval[l];
                    }
                }
            }
        }
        rowIdx = __ldg(&target_row[rowIdx]) * vec_size;
#pragma unroll
        for (int i = 0; i < CF; i++)
        {
            if (i < CFnum)
            {
#pragma unroll
                for (int j = 0; j < vec_size; j++)
                    atomicAdd(&output[rowIdx + j][offset + i * WARP_SIZE + laneId], res[j][i]);
            }
        }
    }
    else
    {
        for (int i = lb; i < hb; i += WARP_SIZE)
        {
            int g_idx = i + laneId;
            if (g_idx < hb)
                col_sh[laneId] = colIdx[g_idx];
            // coaleced access
            scalar_t *val_arr = (scalar_t *)(&values[i][0]);
#pragma unroll
            for (int j = 0; j < vec_size; j++)
                if (j * WARP_SIZE + laneId < (hb - lb) * vec_size)
                    val_sh[j * WARP_SIZE + laneId] = val_arr[j * WARP_SIZE + laneId];
            //             if (g_idx < hb)
            //             {
            //                 col_sh[laneId] = colIdx[g_idx];
            //                 scalar_t *val_arr = (scalar_t *)(&values[i][0]);
            // #pragma unroll
            //                 for (int j = 0; j < vec_size; j++)
            //                     val_sh[j * WARP_SIZE + laneId] = val_arr[j * WARP_SIZE + laneId];
            //                 // val_sh[j * WARP_SIZE + laneId] = values[g_idx][j];
            //             }
            __syncwarp();
            for (int j = 0; j < WARP_SIZE && lb < hb; lb++, j++)
            {
                BrowIdx = col_sh[j];
#pragma unroll
                for (int k = 0; k < vec_size; k++)
                    val[k] = val_sh[k + j * vec_size];
                    // val[k] = val_sh[k * WARP_SIZE + j];
#pragma unroll
                for (int k = 0; k < CF; k++)
                    Bval[k] = input[BrowIdx][offset + k * WARP_SIZE + laneId];

#pragma unroll
                for (int k = 0; k < vec_size; k++)
                {
                    if (val[k] != 0)
                    {
#pragma unroll
                        for (int l = 0; l < CF; l++)
                            res[k][l] += val[k] * Bval[l];
                    }
                }
            }
        }
        rowIdx = __ldg(&target_row[rowIdx]) * vec_size;
#pragma unroll
        for (int i = 0; i < CF; i++)
        {
#pragma unroll
            for (int j = 0; j < vec_size; j++)
                atomicAdd(&output[rowIdx + j][offset + i * WARP_SIZE + laneId], res[j][i]);
        }
    }
}

template <typename scalar_t, int vec_size>
__global__ void vec_split_spmm_cuda_kernel(
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

    scalar_t val[vec_size] = {0};
    scalar_t res[vec_size] = {0};
    scalar_t Bval;

    int lb = __ldg(&rowPtr_bcsr_split[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr_split[rowIdx + 1]);
    if (lb == hb)
        return;
    for (; lb < hb; lb++)
    {
#pragma unroll
        for (int i = 0; i < vec_size; i++)
            val[i] = values[lb][i];
        Bval = input[colIdx[lb]][laneId];
#pragma unroll
        for (int i = 0; i < vec_size; i++)
            if (val[i] != 0)
                res[i] += val[i] * Bval;
    }
    rowIdx = __ldg(&target_row[rowIdx]) * vec_size;
#pragma unroll
    for (int i = 0; i < vec_size; i++)
        atomicAdd(&output[rowIdx + i][laneId], res[i]);
}

template torch::Tensor csr_spmm_cuda<1, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<2, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<3, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<4, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<5, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<6, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<7, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<8, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor csr_spmm_cuda<1, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<2, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<3, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<4, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<5, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<6, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<7, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_spmm_cuda<8, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor csr_split_spmm_cuda<1, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<2, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<3, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<4, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<5, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<6, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<7, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<8, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template torch::Tensor csr_split_spmm_cuda<1, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<2, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<3, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<4, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<5, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<6, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<7, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_spmm_cuda<8, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template torch::Tensor bcsr_spmm_cuda<1, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<2, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<3, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<4, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<5, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<6, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<7, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<8, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor bcsr_spmm_cuda<1, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<2, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<3, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<4, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<5, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<6, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<7, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_spmm_cuda<8, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor bcsr_split_spmm_cuda<1, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<2, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<3, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<4, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<5, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<6, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<7, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<8, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template torch::Tensor bcsr_split_spmm_cuda<1, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<2, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<3, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<4, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<5, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<6, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<7, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_spmm_cuda<8, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template torch::Tensor vec_spmm_cuda<1, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<2, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<3, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<4, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<5, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<6, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<7, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<8, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor vec_spmm_cuda<1, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<2, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<3, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<4, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<5, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<6, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<7, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_spmm_cuda<8, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor vec_split_spmm_cuda<1, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<2, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<3, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<4, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<5, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<6, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<7, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<8, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template torch::Tensor vec_split_spmm_cuda<1, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<2, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<3, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<4, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<5, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<6, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<7, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_spmm_cuda<8, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template int csr_spmm_cuda_profile<1, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<2, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<3, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<4, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<5, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<6, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<7, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<8, 4>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);

template int csr_spmm_cuda_profile<1, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<2, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<3, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<4, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<5, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<6, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<7, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_spmm_cuda_profile<8, 8>(torch::Tensor input, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);

template int csr_split_spmm_cuda_profile<1, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<2, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<3, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<4, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<5, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<6, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<7, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<8, 4>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template int csr_split_spmm_cuda_profile<1, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<2, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<3, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<4, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<5, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<6, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<7, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_spmm_cuda_profile<8, 8>(torch::Tensor input, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template int bcsr_spmm_cuda_profile<1, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<2, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<3, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<4, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<5, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<6, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<7, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<8, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);

template int bcsr_spmm_cuda_profile<1, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<2, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<3, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<4, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<5, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<6, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<7, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_spmm_cuda_profile<8, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);

template int bcsr_split_spmm_cuda_profile<1, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<2, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<3, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<4, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<5, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<6, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<7, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<8, 4>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);

template int bcsr_split_spmm_cuda_profile<1, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<2, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<3, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<4, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<5, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<6, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<7, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_spmm_cuda_profile<8, 8>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);

template int vec_spmm_cuda_profile<1, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<2, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<3, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<4, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<5, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<6, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<7, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<8, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);

template int vec_spmm_cuda_profile<1, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<2, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<3, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<4, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<5, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<6, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<7, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_spmm_cuda_profile<8, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);

template int vec_split_spmm_cuda_profile<1, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<2, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<3, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<4, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<5, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<6, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<7, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<8, 4, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);

template int vec_split_spmm_cuda_profile<1, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<2, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<3, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<4, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<5, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<6, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<7, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_spmm_cuda_profile<8, 8, 3>(torch::Tensor input, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);