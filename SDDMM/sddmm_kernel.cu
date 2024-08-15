#include "sddmm.h"

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
__global__ void csr_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> oval,
    const int nr, const int dim);

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void csr_split_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> oval,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_rows, const int dim);

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void bcsr_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> oval,
    const int nblk_row, const int dim);

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void bcsr_split_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> oval,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim);

template <typename scalar_t, int CF, int blkWarpNum, int vec_size, bool is_residue>
__global__ void vec_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> oval,
    const int nblk_row, const int dim);

template <typename scalar_t, int CF, int blkWarpNum, int vec_size, bool is_residue>
__global__ void vec_split_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> oval,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim);

template <typename scalar_t>
__global__ void genmask_kernel(
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> blkval,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> mask,
    const int nnz, const int n)
{
    // rowIdx = tid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    int lb = __ldg(&rowPtr_bcsr[tid]);
    int hb = __ldg(&rowPtr_bcsr[tid + 1]);
    if (lb == hb)
        return;

    int rowIdx = tid * 2;
    int cnt[2];
    cnt[0] = __ldg(&rowPtr[rowIdx]);
    cnt[1] = __ldg(&rowPtr[rowIdx + 1]);

    for (int i = lb; i < hb; i++)
    {
        if (blkval[i][0] != 0)
            mask[cnt[0]++] = i * 2;
        if (blkval[i][1] != 0)
            mask[cnt[1]++] = i * 2 + 1;    
        
    }
    // for (int i = lb; i < hb; i++)
    // {
    //     if (blkval[i][1] != 0)
    //     {
    //         if (cnt[1] == 541 || cnt[1] == 542 || cnt[1] == 560 || cnt[1] == 561)
    //             printf("tid:%d lb(i):%d rowIdx:%d cnt[1] = %d, mask = %d\n", tid, i, rowIdx + 1, cnt[1], i * 2 + 1);
    //         mask[cnt[1]] = i * 2 + 1;
    //         cnt[1] = cnt[1] + 1;
    //     }
    // }
}

torch::Tensor genmask_cuda(
    torch::Tensor rowPtr_bcsr,
    torch::Tensor rowPtr,
    torch::Tensor blkval,
    int nnz)
{
    auto mask = torch::zeros(nnz, rowPtr.options());
    const int n = rowPtr_bcsr.size(0) - 1;
    const int rowPtr_size = rowPtr.size(0);
    printf("n = %d rowPtr_size = %d\n", n, rowPtr_size);
    dim3 blockDim(256, 1, 1);
    dim3 gridDim(CEIL_DIV(n, blockDim.x), 1, 1);
    AT_DISPATCH_FLOATING_TYPES(blkval.scalar_type(), "genmask_kernel", ([&]
                                                                        { genmask_kernel<scalar_t><<<gridDim, blockDim>>>(
                                                                              rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              blkval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                              mask.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                              nnz,
                                                                              n); }));
    return mask;
}

template <int CF, int blkWarpNum>
torch::Tensor csr_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto oval = torch::zeros_like(values);
    const int num_nodes = rowPtr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                // K
    int dim_x, dim_y;

    dim_x = CEIL_DIV(num_nodes, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    const int smem = blkWarpNum * WARP_SIZE * sizeof(float) * CF;
    if (is_residue)
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "csr_sddmm_cuda", ([&]
                                                                       { csr_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                             A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                             B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                             rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                             colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                             values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             oval.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             num_nodes,
                                                                             dim); }));
    else
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "csr_sddmm_cuda", ([&]
                                                                       { csr_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                             A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                             B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                             rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                             colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                             values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             oval.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                             num_nodes,
                                                                             dim); }));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return oval;
}

template <int CF, int blkWarpNum>
torch::Tensor csr_split_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    auto oval = torch::zeros_like(values);
    const int split_rows = rowPtr_split.size(0) - 1; // number of splitted rows
    const int dim = A.size(1);                       // K
    assert(target_row.size(0) == split_rows);
    int dim_x, dim_y;
    dim_x = CEIL_DIV(split_rows, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(float) * CF;
    if (is_residue)
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "csr_split_sddmm_cuda", ([&]
                                                                             { csr_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                                   A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                   B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                   rowPtr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                   colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                   values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                   oval.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                   target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                   split_rows,
                                                                                   dim); }));
    else
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "csr_split_sddmm_cuda", ([&]
                                                                             { csr_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                                   A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                   B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                   rowPtr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                   colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                   values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                   oval.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                   target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                   split_rows,
                                                                                   dim); }));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return oval;
}

template <int CF, int blkWarpNum>
torch::Tensor bcsr_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto oval = torch::zeros_like(values);
    const int nblk_row = rowPtr_bcsr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                    // embedding size
    int dim_x, dim_y;

    dim_x = CEIL_DIV(nblk_row, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    const int smem = blkWarpNum * WARP_SIZE * sizeof(float) * 2 * CF;
    if (is_residue)
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                           { bcsr_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                                 A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 nblk_row,
                                                                                 dim); }));
    else
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                           { bcsr_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                                 A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 nblk_row,
                                                                                 dim); }));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return oval;
}

template <int CF, int blkWarpNum>
torch::Tensor bcsr_split_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    auto oval = torch::zeros_like(values);
    const int split_nblk_row = rowPtr_bcsr_split.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                                // embedding size
    int dim_x, dim_y;

    dim_x = CEIL_DIV(split_nblk_row, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(float) * CF * 2;
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    if (is_residue)
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                           { bcsr_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                                 A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 split_nblk_row,
                                                                                 dim); }));
    else
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                           { bcsr_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                                 A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 split_nblk_row,
                                                                                 dim); }));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return oval;
}

template <int CF, int blkWarpNum, int vec_size>
torch::Tensor vec_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto oval = torch::zeros_like(values);
    const int nblk_row = rowPtr_bcsr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                    // embedding size

    int dim_x, dim_y;
    dim_x = CEIL_DIV(nblk_row, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * vec_size * CF;
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    if (is_residue)
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                           { vec_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, true><<<gridDim, blockDim, smem>>>(
                                                                                 A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 nblk_row,
                                                                                 dim); }));
    else
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                           { vec_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, false><<<gridDim, blockDim, smem>>>(
                                                                                 A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 nblk_row,
                                                                                 dim); }));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return oval;
}

template <int CF, int blkWarpNum, int vec_size>
torch::Tensor vec_split_sddmm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    auto oval = torch::zeros_like(values);
    const int split_nblk_row = rowPtr_bcsr_split.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                                // embedding size
    int dim_x, dim_y;

    dim_x = CEIL_DIV(split_nblk_row, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(float) * vec_size * CF;
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    if (is_residue)
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                           { vec_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, true><<<gridDim, blockDim, smem>>>(
                                                                                 A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 split_nblk_row,
                                                                                 dim); }));
    else
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                           { vec_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, false><<<gridDim, blockDim, smem>>>(
                                                                                 A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                 target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                 split_nblk_row,
                                                                                 dim); }));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return oval;
}

template <int CF, int blkWarpNum>
int csr_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto oval = torch::zeros_like(values);
    const int num_nodes = rowPtr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    dim_x = CEIL_DIV(num_nodes, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(float) * CF;
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    cudaEventRecord(start, 0);
    if (is_residue)
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "csr-sddmm", ([&]
                                                                      { csr_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                            A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                            B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                            rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                            colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                            values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            oval.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            num_nodes,
                                                                            dim); }));
        }
    else
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "csr-sddmm", ([&]
                                                                      { csr_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                            A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                            B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                            rowPtr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                            colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                            values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            oval.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                            num_nodes,
                                                                            dim); }));
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
int csr_split_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row)
{
    auto oval = torch::zeros_like(values);
    const int split_rows = rowPtr_split.size(0) - 1; // number of splitted rows
    const int dim = A.size(1);                       // embedding size
    assert(target_row.size(0) == split_rows);
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    dim_x = CEIL_DIV(split_rows, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(float) * CF;
    cudaEventRecord(start, 0);
    if (is_residue)
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { csr_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                     target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     split_rows,
                                                                                     dim); }));
        }
    else
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { csr_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                     target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     split_rows,
                                                                                     dim); }));
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
int bcsr_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz)
{
    auto oval = torch::zeros_like(values);
    const int nblk_row = rowPtr_bcsr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                    // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    dim_x = CEIL_DIV(nblk_row, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * 2 * CF;
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    cudaEventRecord(start, 0);
    if (is_residue)
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { bcsr_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     nblk_row,
                                                                                     dim); }));
        }
    else
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { bcsr_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     nblk_row,
                                                                                     dim); }));
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
int bcsr_split_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz)
{
    auto oval = torch::zeros_like(values);
    const int split_nblk_row = rowPtr_bcsr_split.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                                // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    dim_x = CEIL_DIV(split_nblk_row, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * CF * 2;
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    cudaEventRecord(start, 0);
    if (is_residue)
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { bcsr_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, true><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     split_nblk_row,
                                                                                     dim); }));
        }
    else
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { bcsr_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, false><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     split_nblk_row,
                                                                                     dim); }));
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
int vec_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr,
    torch::Tensor colIdx,
    torch::Tensor values,
    int nnz)
{
    auto oval = torch::zeros_like(values);
    const int nblk_row = rowPtr_bcsr.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                    // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    dim_x = CEIL_DIV(nblk_row, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(float) * vec_size * CF;
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    cudaEventRecord(start, 0);
    if (is_residue)
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { vec_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, true><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     nblk_row,
                                                                                     dim); }));
        }
    else
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { vec_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, false><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_bcsr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     nblk_row,
                                                                                     dim); }));
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
int vec_split_sddmm_cuda_profile(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor rowPtr_bcsr_split,
    torch::Tensor colIdx,
    torch::Tensor values,
    torch::Tensor target_row,
    int nnz)
{
    auto oval = torch::zeros_like(values);
    const int split_nblk_row = rowPtr_bcsr_split.size(0) - 1; // number of nodes(rows of matrix)
    const int dim = A.size(1);                                // embedding size
    int dim_x, dim_y;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }

    dim_x = CEIL_DIV(split_nblk_row, blkWarpNum);
    dim_y = CEIL_DIV(dim, CF * WARP_SIZE);
    dim3 gridDim(dim_x, dim_y, 1);
    dim3 blockDim(blkWarpNum * WARP_SIZE, 1, 1);
    const int smem = blkWarpNum * WARP_SIZE * sizeof(int) * vec_size * CF;
    const bool is_residue = dim % (CF * WARP_SIZE) != 0;
    cudaEventRecord(start, 0);
    if (is_residue)
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { vec_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, true><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     split_nblk_row,
                                                                                     dim); }));
        }

    else
        for (int i = 0; i < PROFILE; i++)
        {
            AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "Scatter_and_Gather", ([&]
                                                                               { vec_split_sddmm_cuda_kernel<scalar_t, CF, blkWarpNum, vec_size, false><<<gridDim, blockDim, smem>>>(
                                                                                     A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     rowPtr_bcsr_split.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     colIdx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     oval.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                                     target_row.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                                     split_nblk_row,
                                                                                     dim); }));
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
__global__ void csr_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B, // B transpose
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> oval,
    const int nr, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    extern __shared__ float shared_mem[];
    float *sh_A = &shared_mem[(warpId << 5) * CF]; // local shared memory array of a warp
    if (rowIdx >= nr)
        return;
    scalar_t res = 0;
    int BrowIdx;
    int lb = __ldg(&rowPtr[rowIdx]);
    int hb = __ldg(&rowPtr[rowIdx + 1]);
    if (lb == hb)
        return;

    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId, WARP_SIZE);
#pragma unroll
        for (int i = 0; i < CF; i++)
            if (i < CFnum)
                sh_A[laneId + i * WARP_SIZE] = A[rowIdx][i * WARP_SIZE + laneId];
        for (; lb < hb; lb++)
        {
            res = 0;
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                if (i < CFnum)
                    res += sh_A[laneId + i * WARP_SIZE] * B[BrowIdx][i * WARP_SIZE + laneId];

            for (int stride = 16; stride > 0; stride >>= 1)
                res += __shfl_xor_sync(0xffffffff, res, stride, 32);
            if (laneId == 0)
                oval[lb] = res;
        }
    }
    else
    {
        // prefetch A to shared memory
#pragma unroll
        for (int i = 0; i < CF; i++)
            sh_A[laneId + i * WARP_SIZE] = A[rowIdx][i * WARP_SIZE + laneId];
        for (; lb < hb; lb++)
        {
            res = 0;
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                res += sh_A[laneId + i * WARP_SIZE] * B[BrowIdx][i * WARP_SIZE + laneId];

            for (int stride = 16; stride > 0; stride >>= 1)
                res += __shfl_xor_sync(0xffffffff, res, stride, 32);
            if (laneId == 0)
                oval[lb] = res;
        }
    }
}

// TODO: extra shared memory for val is not needed
template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void csr_split_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> oval,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_rows, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    extern __shared__ float shared_mem[];
    float *sh_A = &shared_mem[(warpId << 5) * CF];
    if (rowIdx >= split_rows)
        return;

    scalar_t res = 0;
    int BrowIdx;
    int lb = __ldg(&rowPtr_split[rowIdx]);
    int hb = __ldg(&rowPtr_split[rowIdx + 1]);
    rowIdx = __ldg(&target_row[rowIdx]);

    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId, WARP_SIZE);
#pragma unroll
        for (int i = 0; i < CF; i++)
            if (i < CFnum)
                sh_A[laneId + i * WARP_SIZE] = A[rowIdx][i * WARP_SIZE + laneId];
        for (; lb < hb; lb++)
        {
            res = 0;
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                if (i < CFnum)
                    res += sh_A[laneId + i * WARP_SIZE] * B[BrowIdx][i * WARP_SIZE + laneId];

            for (int stride = 16; stride > 0; stride >>= 1)
                res += __shfl_xor_sync(0xffffffff, res, stride, 32);
            if (laneId == 0)
                oval[lb] = res;
        }
    }
    else
    {
        // prefetch A to shared memory
#pragma unroll
        for (int i = 0; i < CF; i++)
            sh_A[laneId + i * WARP_SIZE] = A[rowIdx][i * WARP_SIZE + laneId];
        for (; lb < hb; lb++)
        {
            res = 0;
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                res += sh_A[laneId + i * WARP_SIZE] * B[BrowIdx][i * WARP_SIZE + laneId];

            for (int stride = 16; stride > 0; stride >>= 1)
                res += __shfl_xor_sync(0xffffffff, res, stride, 32);
            if (laneId == 0)
            {
                oval[lb] = res;
                // if (lb==18)
                // printf("rowIdx:%d, lb:%d, warpId:%d, res:%f\n", rowIdx, lb, warpId, res);
            }
        }
    }
}

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void bcsr_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> oval,
    const int nblk_row, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    extern __shared__ float shared_mem[];
    float *sh_A = &shared_mem[(warpId << 5) * CF * 2];
    if (rowIdx >= nblk_row)
        return;

    scalar_t res[2] = {0};
    scalar_t Bval[CF] = {0};
    int BrowIdx;
    int lb = __ldg(&rowPtr_bcsr[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr[rowIdx + 1]);
    rowIdx *= 2;
    if (lb == hb)
        return;

    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId, WARP_SIZE);
#pragma unroll
        for (int i = 0; i < CF; i++)
            if (i < CFnum)
            {
                sh_A[laneId + i * WARP_SIZE] = A[rowIdx][i * WARP_SIZE + laneId];
                sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF] = A[rowIdx + 1][i * WARP_SIZE + laneId];
            }
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                Bval[i] = B[BrowIdx][i * WARP_SIZE + laneId];

            if (values[lb][0] != 0)
            {
                res[0] = 0;
#pragma unroll
                for (int i = 0; i < CF; i++)
                    if (i < CFnum)
                        res[0] += sh_A[laneId + i * WARP_SIZE] * Bval[i];
                for (int stride = 16; stride > 0; stride >>= 1)
                    res[0] += __shfl_xor_sync(0xffffffff, res[0], stride, 32);
                if (laneId == 0)
                    oval[lb][0] = res[0];
            }
            if (values[lb][1] != 0)
            {
                res[1] = 0;
#pragma unroll
                for (int i = 0; i < CF; i++)
                    if (i < CFnum)
                        res[1] += sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF] * Bval[i];
                for (int stride = 16; stride > 0; stride >>= 1)
                    res[1] += __shfl_xor_sync(0xffffffff, res[1], stride, 32);
                if (laneId == 0)
                    oval[lb][1] = res[1];
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < CF; i++)
        {
            sh_A[laneId + i * WARP_SIZE] = A[rowIdx][i * WARP_SIZE + laneId];
            sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF] = A[rowIdx + 1][i * WARP_SIZE + laneId];
        }
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                Bval[i] = B[BrowIdx][i * WARP_SIZE + laneId];

            if (values[lb][0] != 0)
            {
                res[0] = 0;
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[0] += sh_A[laneId + i * WARP_SIZE] * Bval[i];
                for (int stride = 16; stride > 0; stride >>= 1)
                    res[0] += __shfl_xor_sync(0xffffffff, res[0], stride, 32);
                if (laneId == 0)
                {
                    oval[lb][0] = res[0];
                    // if (rowIdx == 70)
                    //     printf("0-rowIdx:%d, lb:%d, warpId:%d, res:%f\n", rowIdx, lb, warpId, res[0]);
                }
            }
            if (values[lb][1] != 0)
            {
                res[1] = 0;
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[1] += sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF] * Bval[i];
                for (int stride = 16; stride > 0; stride >>= 1)
                    res[1] += __shfl_xor_sync(0xffffffff, res[1], stride, 32);
                if (laneId == 0)
                {
                    oval[lb][1] = res[1];
                    // if (rowIdx == 70)
                    //     printf("1-rowIdx:%d, lb:%d, warpId:%d, res:%f\n", rowIdx, lb, warpId, res[1]);
                }
            }
        }
    }
}

template <typename scalar_t, int CF, int blkWarpNum, bool is_residue>
__global__ void bcsr_split_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> oval,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    extern __shared__ float shared_mem[];
    float *sh_A = &shared_mem[(warpId << 5) * CF * 2];
    if (rowIdx >= split_nblk_row)
        return;

    scalar_t res[2] = {0};
    scalar_t Bval[CF] = {0};
    int BrowIdx;
    int lb = __ldg(&rowPtr_bcsr_split[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr_split[rowIdx + 1]);
    rowIdx = __ldg(&target_row[rowIdx]) * 2;

    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId, WARP_SIZE);
#pragma unroll
        for (int i = 0; i < CF; i++)
            if (i < CFnum)
            {
                sh_A[laneId + i * WARP_SIZE] = A[rowIdx][i * WARP_SIZE + laneId];
                sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF] = A[rowIdx + 1][i * WARP_SIZE + laneId];
            }
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                if (i < CFnum)
                    Bval[i] = B[BrowIdx][i * WARP_SIZE + laneId];

            if (values[lb][0] != 0)
            {
                res[0] = 0;
#pragma unroll
                for (int i = 0; i < CF; i++)
                    if (i < CFnum)
                        res[0] += sh_A[laneId + i * WARP_SIZE] * Bval[i];
                for (int stride = 16; stride > 0; stride >>= 1)
                    res[0] += __shfl_xor_sync(0xffffffff, res[0], stride, 32);
                if (laneId == 0)
                    oval[lb][0] = res[0];
            }
            if (values[lb][1] != 0)
            {
                res[1] = 0;
#pragma unroll
                for (int i = 0; i < CF; i++)
                    if (i < CFnum)
                        res[1] += sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF] * Bval[i];
                for (int stride = 16; stride > 0; stride >>= 1)
                    res[1] += __shfl_xor_sync(0xffffffff, res[1], stride, 32);
                if (laneId == 0)
                    oval[lb][1] = res[1];
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < CF; i++)
        {
            sh_A[laneId + i * WARP_SIZE] = A[rowIdx][i * WARP_SIZE + laneId];
            sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF] = A[rowIdx + 1][i * WARP_SIZE + laneId];
        }
        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                Bval[i] = B[BrowIdx][i * WARP_SIZE + laneId];

            if (values[lb][0] != 0)
            {
                res[0] = 0;
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[0] += sh_A[laneId + i * WARP_SIZE] * Bval[i];
                for (int stride = 16; stride > 0; stride >>= 1)
                    res[0] += __shfl_xor_sync(0xffffffff, res[0], stride, 32);
                if (laneId == 0)
                    oval[lb][0] = res[0];
            }
            if (values[lb][1] != 0)
            {
                res[1] = 0;
#pragma unroll
                for (int i = 0; i < CF; i++)
                    res[1] += sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF] * Bval[i];
                for (int stride = 16; stride > 0; stride >>= 1)
                    res[1] += __shfl_xor_sync(0xffffffff, res[1], stride, 32);
                if (laneId == 0)
                    oval[lb][1] = res[1];
            }
        }
    }
}

template <typename scalar_t, int CF, int blkWarpNum, int vec_size, bool is_residue>
__global__ void vec_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> oval,
    const int nblk_row, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    extern __shared__ float shared_mem[];
    float *sh_A = &shared_mem[(warpId << 5) * CF * 2];
    if (rowIdx >= nblk_row)
        return;

    scalar_t res[vec_size] = {0};
    scalar_t Bval[CF] = {0};
    int BrowIdx;
    int lb = __ldg(&rowPtr_bcsr[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr[rowIdx + 1]);
    rowIdx *= vec_size;
    if (lb == hb)
        return;

    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId, WARP_SIZE);
#pragma unroll
        for (int i = 0; i < CF; i++)
            if (i < CFnum)
                for (int j = 0; j < vec_size; j++)
                    sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF * j] = A[rowIdx + j][i * WARP_SIZE + laneId];

        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                if (i < CFnum)
                    Bval[i] = B[BrowIdx][i * WARP_SIZE + laneId];
            for (int i = 0; i < vec_size; i++)
            {
                if (values[lb][i] != 0)
                {
                    res[i] = 0;
#pragma unroll
                    for (int j = 0; j < CF; j++)
                        if (i < CFnum)
                            res[i] += sh_A[laneId + j * WARP_SIZE + WARP_SIZE * CF * i] * Bval[j];
                    for (int stride = 16; stride > 0; stride >>= 1)
                        res[i] += __shfl_xor_sync(0xffffffff, res[i], stride, 32);
                    if (laneId == 0)
                        oval[lb][i] = res[i];
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < CF; i++)
            for (int j = 0; j < vec_size; j++)
                sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF * j] = A[rowIdx + j][i * WARP_SIZE + laneId];

        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                Bval[i] = B[BrowIdx][i * WARP_SIZE + laneId];
            for (int i = 0; i < vec_size; i++)
            {
                if (values[lb][i] != 0)
                {
                    res[i] = 0;
#pragma unroll
                    for (int j = 0; j < CF; j++)
                        res[i] += sh_A[laneId + j * WARP_SIZE + WARP_SIZE * CF * i] * Bval[j];
                    for (int stride = 16; stride > 0; stride >>= 1)
                        res[i] += __shfl_xor_sync(0xffffffff, res[i], stride, 32);
                    if (laneId == 0)
                        oval[lb][i] = res[i];
                }
            }
        }
    }
}

template <typename scalar_t, int CF, int blkWarpNum, int vec_size, bool is_residue>
__global__ void vec_split_sddmm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rowPtr_bcsr_split,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> colIdx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> oval,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> target_row,
    const int split_nblk_row, const int dim)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int rowIdx = blockIdx.x * blkWarpNum + warpId;
    extern __shared__ float shared_mem[];
    float *sh_A = &shared_mem[(warpId << 5) * CF * 2];
    if (rowIdx >= split_nblk_row)
        return;

    scalar_t res[vec_size] = {0};
    scalar_t Bval[CF] = {0};
    int BrowIdx;
    int lb = __ldg(&rowPtr_bcsr_split[rowIdx]);
    int hb = __ldg(&rowPtr_bcsr_split[rowIdx + 1]);
    rowIdx = __ldg(&target_row[rowIdx]) * vec_size;

    if (is_residue)
    {
        int CFnum = CEIL_DIV(dim - laneId, WARP_SIZE);
#pragma unroll
        for (int i = 0; i < CF; i++)
            if (i < CFnum)
                for (int j = 0; j < vec_size; j++)
                    sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF * j] = A[rowIdx + j][i * WARP_SIZE + laneId];

        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                if (i < CFnum)
                    Bval[i] = B[BrowIdx][i * WARP_SIZE + laneId];
            for (int i = 0; i < vec_size; i++)
            {
                if (values[lb][i] != 0)
                {
                    res[i] = 0;
#pragma unroll
                    for (int j = 0; j < CF; j++)
                        if (i < CFnum)
                            res[i] += sh_A[laneId + j * WARP_SIZE + WARP_SIZE * CF * i] * Bval[j];
                    for (int stride = 16; stride > 0; stride >>= 1)
                        res[i] += __shfl_xor_sync(0xffffffff, res[i], stride, 32);
                    if (laneId == 0)
                        oval[lb][i] = res[i];
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < CF; i++)
            for (int j = 0; j < vec_size; j++)
                sh_A[laneId + i * WARP_SIZE + WARP_SIZE * CF * j] = A[rowIdx + j][i * WARP_SIZE + laneId];

        for (; lb < hb; lb++)
        {
            BrowIdx = colIdx[lb];
#pragma unroll
            for (int i = 0; i < CF; i++)
                Bval[i] = B[BrowIdx][i * WARP_SIZE + laneId];
            for (int i = 0; i < vec_size; i++)
            {
                if (values[lb][i] != 0)
                {
                    res[i] = 0;
#pragma unroll
                    for (int j = 0; j < CF; j++)
                        res[i] += sh_A[laneId + j * WARP_SIZE + WARP_SIZE * CF * i] * Bval[j];
                    for (int stride = 16; stride > 0; stride >>= 1)
                        res[i] += __shfl_xor_sync(0xffffffff, res[i], stride, 32);
                    if (laneId == 0)
                        oval[lb][i] = res[i];
                }
            }
        }
    }
}

template torch::Tensor csr_sddmm_cuda<1, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_sddmm_cuda<2, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_sddmm_cuda<3, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_sddmm_cuda<4, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_sddmm_cuda<1, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_sddmm_cuda<2, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_sddmm_cuda<3, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor csr_sddmm_cuda<4, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor csr_split_sddmm_cuda<1, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_sddmm_cuda<2, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_sddmm_cuda<3, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_sddmm_cuda<4, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_sddmm_cuda<1, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_sddmm_cuda<2, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_sddmm_cuda<3, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor csr_split_sddmm_cuda<4, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template torch::Tensor bcsr_sddmm_cuda<1, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_sddmm_cuda<2, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_sddmm_cuda<3, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_sddmm_cuda<4, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_sddmm_cuda<1, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_sddmm_cuda<2, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_sddmm_cuda<3, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor bcsr_sddmm_cuda<4, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor bcsr_split_sddmm_cuda<1, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_sddmm_cuda<2, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_sddmm_cuda<3, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_sddmm_cuda<4, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_sddmm_cuda<1, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_sddmm_cuda<2, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_sddmm_cuda<3, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor bcsr_split_sddmm_cuda<4, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template torch::Tensor vec_sddmm_cuda<1, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_sddmm_cuda<2, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_sddmm_cuda<3, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_sddmm_cuda<4, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_sddmm_cuda<1, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_sddmm_cuda<2, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_sddmm_cuda<3, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);
template torch::Tensor vec_sddmm_cuda<4, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values);

template torch::Tensor vec_split_sddmm_cuda<1, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_sddmm_cuda<2, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_sddmm_cuda<3, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_sddmm_cuda<4, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_sddmm_cuda<1, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_sddmm_cuda<2, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_sddmm_cuda<3, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template torch::Tensor vec_split_sddmm_cuda<4, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template int csr_sddmm_cuda_profile<1, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_sddmm_cuda_profile<2, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_sddmm_cuda_profile<3, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_sddmm_cuda_profile<4, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_sddmm_cuda_profile<1, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_sddmm_cuda_profile<2, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_sddmm_cuda_profile<3, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);
template int csr_sddmm_cuda_profile<4, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr, torch::Tensor colIdx, torch::Tensor values);

template int csr_split_sddmm_cuda_profile<1, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_sddmm_cuda_profile<2, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_sddmm_cuda_profile<3, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_sddmm_cuda_profile<4, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_sddmm_cuda_profile<1, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_sddmm_cuda_profile<2, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_sddmm_cuda_profile<3, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);
template int csr_split_sddmm_cuda_profile<4, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row);

template int bcsr_sddmm_cuda_profile<1, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_sddmm_cuda_profile<2, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_sddmm_cuda_profile<3, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_sddmm_cuda_profile<4, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_sddmm_cuda_profile<1, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_sddmm_cuda_profile<2, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_sddmm_cuda_profile<3, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int bcsr_sddmm_cuda_profile<4, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);

template int bcsr_split_sddmm_cuda_profile<1, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_sddmm_cuda_profile<2, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_sddmm_cuda_profile<3, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_sddmm_cuda_profile<4, 4>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_sddmm_cuda_profile<1, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_sddmm_cuda_profile<2, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_sddmm_cuda_profile<3, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int bcsr_split_sddmm_cuda_profile<4, 8>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);

template int vec_sddmm_cuda_profile<1, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_sddmm_cuda_profile<2, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_sddmm_cuda_profile<3, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_sddmm_cuda_profile<4, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_sddmm_cuda_profile<1, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_sddmm_cuda_profile<2, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_sddmm_cuda_profile<3, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);
template int vec_sddmm_cuda_profile<4, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr, torch::Tensor colIdx, torch::Tensor values, int nnz);

template int vec_split_sddmm_cuda_profile<1, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_sddmm_cuda_profile<2, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_sddmm_cuda_profile<3, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_sddmm_cuda_profile<4, 4, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_sddmm_cuda_profile<1, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_sddmm_cuda_profile<2, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_sddmm_cuda_profile<3, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
template int vec_split_sddmm_cuda_profile<4, 8, 3>(torch::Tensor A, torch::Tensor B, torch::Tensor rowPtr_bcsr_split, torch::Tensor colIdx, torch::Tensor values, torch::Tensor target_row, int nnz);
