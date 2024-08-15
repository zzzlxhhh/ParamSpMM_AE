#include "SpMM.h"
#define CUSPARSE_CHECK(func)                                                         \
    {                                                                                \
        cusparseStatus_t status = (func);                                            \
        if (status != CUSPARSE_STATUS_SUCCESS)                                       \
        {                                                                            \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
                   cusparseGetErrorString(status), status);                          \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    }
#define CUDA_CHECK(func)                                                                           \
    {                                                                                              \
        cudaError_t status = (func);                                                               \
        if (status != cudaSuccess)                                                                 \
        {                                                                                          \
            printf("CUDA API failed at file %s line %d with error: %s (%d)\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status), status);                                            \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

// vendor-provided cusparse by NVIDIA
torch::Tensor cusparse_compute(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);    // row of A
    const int K = M;                // col of A
    const int N = output.size(1);   // col of B
    const int nnz = values.size(0); // nnz of A

    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrDescr;
    cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    auto csr_indptr_d = rowPtr.data_ptr<int>();
    auto csr_indices_d = colIdx.data_ptr<int>();
    auto csr_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    // creating sparse csr matrix
    CUSPARSE_CHECK(cusparseCreateCsr(
        &csrDescr, M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr, M, N, N, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t workspace_size;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
        &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &workspace_size));

    void *workspace = NULL;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    // run SpMM
    CUSPARSE_CHECK(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                &alpha, csrDescr, dnMatInputDescr, &beta,
                                dnMatOutputDescr, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, workspace));
    return output;
}

// vendor-provided cusparse by NVIDIA
int cusparse_profile(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);    // row of A
    const int K = M;                // col of A
    const int N = output.size(1);   // col of B
    const int nnz = values.size(0); // nnz of A

    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrDescr;
    cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    auto csr_indptr_d = rowPtr.data_ptr<int>();
    auto csr_indices_d = colIdx.data_ptr<int>();
    auto csr_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    // creating sparse csr matrix
    CUSPARSE_CHECK(cusparseCreateCsr(
        &csrDescr, M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr, M, N, N, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t workspace_size;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
        &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &workspace_size));

    void *workspace = NULL;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }
    cudaEventRecord(start, 0);
    for (int i = 0; i < PROFILE; i++)
    {
        // run SpMM
        CUSPARSE_CHECK(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                    &alpha, csrDescr, dnMatInputDescr, &beta,
                                    dnMatOutputDescr, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, workspace));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = nnz / 1e6 * N * 2;
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

// vendor-provided CUSPARSE_SPMM_CSR_ALG2 by NVIDIA
torch::Tensor cusparse_compute_csr_alg2(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);    // row of A
    const int K = M;                // col of A
    const int N = output.size(1);   // col of B
    const int nnz = values.size(0); // nnz of A

    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrDescr;
    cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    auto csr_indptr_d = rowPtr.data_ptr<int>();
    auto csr_indices_d = colIdx.data_ptr<int>();
    auto csr_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    // creating sparse csr matrix
    CUSPARSE_CHECK(cusparseCreateCsr(
        &csrDescr, M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr, M, N, N, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t workspace_size;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
        &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
        &workspace_size));

    void *workspace = NULL;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    // run SpMM
    CUSPARSE_CHECK(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                &alpha, csrDescr, dnMatInputDescr, &beta,
                                dnMatOutputDescr, CUDA_R_32F,
                                CUSPARSE_SPMM_CSR_ALG2, workspace));
    return output;
}

// vendor-provided CUSPARSE_SPMM_CSR_ALG2 by NVIDIA
int cusparse_profile_csr_alg2(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);    // row of A
    const int K = M;                // col of A
    const int N = output.size(1);   // col of B
    const int nnz = values.size(0); // nnz of A

    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrDescr;
    cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    auto csr_indptr_d = rowPtr.data_ptr<int>();
    auto csr_indices_d = colIdx.data_ptr<int>();
    auto csr_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    // creating sparse csr matrix
    CUSPARSE_CHECK(cusparseCreateCsr(
        &csrDescr, M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr, M, N, N, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t workspace_size;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
        &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
        &workspace_size));

    void *workspace = NULL;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }
    cudaEventRecord(start, 0);
    for (int i = 0; i < PROFILE; i++)
    {
        // run SpMM
        CUSPARSE_CHECK(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                    &alpha, csrDescr, dnMatInputDescr, &beta,
                                    dnMatOutputDescr, CUDA_R_32F,
                                    CUSPARSE_SPMM_CSR_ALG2, workspace));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = nnz / 1e6 * N * 2;
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

torch::Tensor cusparse_compute_coo_alg4(
    torch::Tensor input,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);      // row of A
    const int K = M;                  // col of A
    const int N = output.size(1);     // col of B
    const int A_nnz = values.size(0); // nnz of A

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle))

    auto coo_row_d = row.data_ptr<int>();
    auto coo_col_d = col.data_ptr<int>();
    auto coo_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    //  Create sparse matrix A in COO format
    CUSPARSE_CHECK(cusparseCreateCoo(&matA, M, K, A_nnz,
                                     coo_row_d, coo_col_d, coo_values_d,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, M, N, N, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, M, N, N, C_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate an external buffer if needed
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize))

    // execute SpMM
    CUSPARSE_CHECK(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))
    return output;
}

int cusparse_profile_coo_alg4(
    torch::Tensor input,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);      // row of A
    const int K = M;                  // col of A
    const int N = output.size(1);     // col of B
    const int A_nnz = values.size(0); // nnz of A

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle))

    auto coo_row_d = row.data_ptr<int>();
    auto coo_col_d = col.data_ptr<int>();
    auto coo_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    //  Create sparse matrix A in COO format
    CUSPARSE_CHECK(cusparseCreateCoo(&matA, M, K, A_nnz,
                                     coo_row_d, coo_col_d, coo_values_d,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, M, N, N, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, M, N, N, C_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate an external buffer if needed
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize))

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warm up
    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }
    cudaEventRecord(start, 0);
    for (int i = 0; i < PROFILE; i++)
    {
        // execute SpMM
        CUSPARSE_CHECK(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = A_nnz / 1e6 * N * 2;
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

torch::Tensor cusparse_compute_csc(
    torch::Tensor input,
    torch::Tensor colPtr,
    torch::Tensor rowIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);    // row of A
    const int K = M;                // col of A
    const int N = output.size(1);   // col of B
    const int nnz = values.size(0); // nnz of A

    cusparseHandle_t handle;
    cusparseSpMatDescr_t cscDescr;
    cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    auto csc_indptr_d = colPtr.data_ptr<int>();
    auto csc_indices_d = rowIdx.data_ptr<int>();
    auto csc_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    // creating sparse csc matrix
    CUSPARSE_CHECK(cusparseCreateCsc(
        &cscDescr, M, K, nnz, csc_indptr_d, csc_indices_d, csc_values_d,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr, M, N, N, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t workspace_size;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cscDescr, dnMatInputDescr,
        &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &workspace_size));

    void *workspace = NULL;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    // run SpMM
    CUSPARSE_CHECK(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                &alpha, cscDescr, dnMatInputDescr, &beta,
                                dnMatOutputDescr, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, workspace));
   return output;
}


int cusparse_profile_csc(
    torch::Tensor input,
    torch::Tensor colPtr,
    torch::Tensor rowIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);    // row of A
    const int K = M;                // col of A
    const int N = output.size(1);   // col of B
    const int nnz = values.size(0); // nnz of A

    cusparseHandle_t handle;
    cusparseSpMatDescr_t cscDescr;
    cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    auto csc_indptr_d = colPtr.data_ptr<int>();
    auto csc_indices_d = rowIdx.data_ptr<int>();
    auto csc_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    // creating sparse csc matrix
    CUSPARSE_CHECK(cusparseCreateCsc(
        &cscDescr, M, K, nnz, csc_indptr_d, csc_indices_d, csc_values_d,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr, M, N, N, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t workspace_size;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cscDescr, dnMatInputDescr,
        &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &workspace_size));

    void *workspace = NULL;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }
    cudaEventRecord(start, 0);
    for (int i = 0; i < PROFILE; i++)
    {
        // run SpMM
        CUSPARSE_CHECK(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                    &alpha, cscDescr, dnMatInputDescr, &beta,
                                    dnMatOutputDescr, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, workspace));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = nnz / 1e6 * N * 2;
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

/*
static SpMM by GeSpMM open source code from
https://github.com/dgSPARSE/dgSPARSE-Lib/tree/main/src/ge-spmm
*/
constexpr int RefThreadPerBlock = 256;
template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset)
{
    if (base != nullptr)
        return base[offset];
    else
        return static_cast<T>(1);
}

template <int CoarsenFactor>
__global__ void csrspmm_rowcaching_rowbalance_kernel(
    const int M, const int N, const int K, const int csr_indptr[],
    const int csr_indices[], const float csr_data[], const float B[],
    float C[])
{
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    extern __shared__ int shared_mem[];
    int *workspace_indices = &shared_mem[(warp_id << 5)];
    float *workspace_data =
        (float *)(workspace_indices +
                  blockDim.x); // float and int has the same size

    // get the sparse-value range of this row
    int row_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (row_id >= M)
        return;
    int start = csr_indptr[row_id];
    int end = csr_indptr[row_id + 1];

    // get the dense column offset
    int col_offset = blockIdx.y * 32 * CoarsenFactor;
    const float *B_lanes[CoarsenFactor];
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++)
    {
        B_lanes[i] = B + col_offset + lane_id + i * 32;
    }
    int ldB = N;

    // declare accumulators
    float c[CoarsenFactor] = {0.0f};
    int ldC = N;

    // N-dimension residual handling
    if (blockIdx.y == gridDim.y - 1)
        goto Ndim_Residue;

    // iterate over the sparse row
    for (int p = start; p < end; p += 32)
    {
        // copy a bucket of sparse row elements into shared memory
        if (p + lane_id < end)
        {
            workspace_data[lane_id] =
                __guard_load_default_one<float>(csr_data, (p + lane_id));
            workspace_indices[lane_id] = csr_indices[p + lane_id];
        }
        else
        {
            workspace_data[lane_id] = 0.0f;
            workspace_indices[lane_id] = 0;
        }
        __syncwarp();
// do MAC computation using buffered elements
#pragma unroll
        for (int pp = 0; pp < 32; pp++)
        {
            int k = workspace_indices[pp];
            float v = workspace_data[pp];
#pragma unroll
            for (int i = 0; i < CoarsenFactor; i++)
            {
                c[i] += v * B_lanes[i][k * ldB];
            }
        }
    }

// write results
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++)
    {
        float *C_lane = B_lanes[i] - B + (C + row_id * ldC);
        *C_lane = c[i];
    }
    return;

Ndim_Residue:
    int valid_lane_num = CEIL_DIV(N - col_offset - lane_id, 32);

    // iterate over the sparse row
    for (int p = start; p < end; p += 32)
    {
        // copy a bucket of sparse row elements into shared memory
        if (p + lane_id < end)
        {
            workspace_data[lane_id] =
                __guard_load_default_one<float>(csr_data, (p + lane_id));
            workspace_indices[lane_id] = csr_indices[p + lane_id];
        }
        else
        {
            workspace_data[lane_id] = 0.0f;
            workspace_indices[lane_id] = 0;
        }
        __syncwarp();
// do MAC computation using buffered elements
#pragma unroll
        for (int pp = 0; pp < 32; pp++)
        {
            int k = workspace_indices[pp];
            float v = workspace_data[pp];
#pragma unroll
            for (int i = 0; i < CoarsenFactor; i++)
            {
                if (i < valid_lane_num)
                {
                    c[i] += v * B_lanes[i][k * ldB];
                }
            }
        }
    }

// write results
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++)
    {
        float *C_lane = B_lanes[i] - B + (C + row_id * ldC);
        if (i < valid_lane_num)
        {
            *C_lane = c[i];
        }
    }
    return;
}

torch::Tensor gespmm_compute(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{

    auto output = torch::zeros_like(input);
    const int M = input.size(0);    // row of A
    const int K = M;                // col of A
    const int N = output.size(1);   // col of B
    const int nnz = values.size(0); // nnz of A

    auto csr_indptr_d = rowPtr.data_ptr<int>();
    auto csr_indices_d = colIdx.data_ptr<int>();
    auto csr_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2
                                                     : 1;
    int Ndim_threadblock = CEIL_DIV(N, (32 * coarsen_factor));
    int Mdim_warp_per_tb = RefThreadPerBlock / 32;
    dim3 gridDim(CEIL_DIV(M, Mdim_warp_per_tb), Ndim_threadblock, 1);
    dim3 blockDim(RefThreadPerBlock, 1, 1);
    size_t smem_size = (sizeof(int) + sizeof(float)) * RefThreadPerBlock;

    if (coarsen_factor == 4)
    {
        csrspmm_rowcaching_rowbalance_kernel<4><<<gridDim, blockDim, smem_size>>>(
            M, N, K, csr_indptr_d, csr_indices_d, csr_values_d, B_d, C_d);
    }
    else if (coarsen_factor == 2)
    {
        csrspmm_rowcaching_rowbalance_kernel<2><<<gridDim, blockDim, smem_size>>>(
            M, N, K, csr_indptr_d, csr_indices_d, csr_values_d, B_d, C_d);
    }
    else
    {
        csrspmm_rowcaching_rowbalance_kernel<1><<<gridDim, blockDim, smem_size>>>(
            M, N, K, csr_indptr_d, csr_indices_d, csr_values_d, B_d, C_d);
    }

    return output;
}

int gespmm_profile(
    torch::Tensor input,
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor values)
{
    auto output = torch::zeros_like(input);
    const int M = input.size(0);    // row of A
    const int K = M;                // col of A
    const int N = output.size(1);   // col of B
    const int nnz = values.size(0); // nnz of A

    auto csr_indptr_d = rowPtr.data_ptr<int>();
    auto csr_indices_d = colIdx.data_ptr<int>();
    auto csr_values_d = values.data_ptr<float>();
    auto B_d = input.data_ptr<float>();
    auto C_d = output.data_ptr<float>();

    int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2
                                                     : 1;
    int Ndim_threadblock = CEIL_DIV(N, (32 * coarsen_factor));
    int Mdim_warp_per_tb = RefThreadPerBlock / 32;
    dim3 gridDim(CEIL_DIV(M, Mdim_warp_per_tb), Ndim_threadblock, 1);
    dim3 blockDim(RefThreadPerBlock, 1, 1);
    size_t smem_size = (sizeof(int) + sizeof(float)) * RefThreadPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warmup in GPU
    for (int i = 0; i < PROFILE; i++)
    {
        warmup<<<1024, 256>>>();
    }
    cudaEventRecord(start, 0);
    for (int i = 0; i < PROFILE; i++)
    {
        if (coarsen_factor == 4)
        {
            csrspmm_rowcaching_rowbalance_kernel<4><<<gridDim, blockDim, smem_size>>>(
                M, N, K, csr_indptr_d, csr_indices_d, csr_values_d, B_d, C_d);
        }
        else if (coarsen_factor == 2)
        {
            csrspmm_rowcaching_rowbalance_kernel<2><<<gridDim, blockDim, smem_size>>>(
                M, N, K, csr_indptr_d, csr_indices_d, csr_values_d, B_d, C_d);
        }
        else
        {
            csrspmm_rowcaching_rowbalance_kernel<1><<<gridDim, blockDim, smem_size>>>(
                M, N, K, csr_indptr_d, csr_indices_d, csr_values_d, B_d, C_d);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = nnz / 1e6 * N * 2;
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