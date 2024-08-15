#include "sddmm.h"
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

// vendor-provided cusparse sddmm by NVIDIA
torch::Tensor cusparse_sddmm_compute(
    torch::Tensor A, // A_ M x k
    torch::Tensor B, // B_ transpose N x k
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor &values)
{
    const int M = A.size(0); // row of A
    const int K = A.size(1); // col of A
    const int N = B.size(0); // col of B
    const int nnz = values.size(0); 

    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrDescr;
    cusparseDnMatDescr_t AMatDecsr, BMatDecsr;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // S
    auto csr_indptr_d = rowPtr.data_ptr<int>();
    auto csr_indices_d = colIdx.data_ptr<int>();
    auto csr_values_d = values.data_ptr<float>();
    // A & B
    auto A_d = A.data_ptr<float>();
    auto B_d = B.data_ptr<float>();

    // // creating sparse csr matrix
    CUSPARSE_CHECK(cusparseCreateCsr(
        &csrDescr, M, N, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&AMatDecsr, M, K, K, A_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&BMatDecsr, N, K, K, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t bufferSize = 0;
    void *dBuffer = NULL;
    CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, AMatDecsr, BMatDecsr, &beta, csrDescr, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // run SDDMM
    CUSPARSE_CHECK(cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE, &alpha, AMatDecsr,
                                 BMatDecsr, &beta, csrDescr, CUDA_R_32F,
                                 CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));
    return values;
}

// vendor-provided cusparse by NVIDIA
int cusparse_sddmm_profile(
    torch::Tensor A, // A_ M x k
    torch::Tensor B, // B_ transpose N x k
    torch::Tensor rowPtr,
    torch::Tensor colIdx,
    torch::Tensor &values)
{
    const int M = A.size(0); // row of A
    const int K = A.size(1); // col of A
    const int N = B.size(0); // col of B
    const int nnz = values.size(0); 

    cusparseHandle_t handle;
    cusparseSpMatDescr_t csrDescr;
    cusparseDnMatDescr_t AMatDecsr, BMatDecsr;
    float alpha = 1.0f, beta = 0.0f;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // S
    auto csr_indptr_d = rowPtr.data_ptr<int>();
    auto csr_indices_d = colIdx.data_ptr<int>();
    auto csr_values_d = values.data_ptr<float>();
    // A & B
    auto A_d = A.data_ptr<float>();
    auto B_d = B.data_ptr<float>();

    // // creating sparse csr matrix
    CUSPARSE_CHECK(cusparseCreateCsr(
        &csrDescr, M, N, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
        CUSPARSE_INDEX_32I, // index 32-integer for indptr
        CUSPARSE_INDEX_32I, // index 32-integer for indices
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F // datatype: 32-bit float real number
        ));

    // creating dense matrices
    CUSPARSE_CHECK(cusparseCreateDnMat(&AMatDecsr, M, K, K, A_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&BMatDecsr, N, K, K, B_d, CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));

    // allocate workspace buffer
    size_t bufferSize = 0;
    void *dBuffer = NULL;
    CUSPARSE_CHECK(cusparseSDDMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, AMatDecsr, BMatDecsr, &beta, csrDescr, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

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
        // run SDDMM
        CUSPARSE_CHECK(cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_TRANSPOSE, &alpha, AMatDecsr,
                                     BMatDecsr, &beta, csrDescr, CUDA_R_32F,
                                     CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = (float)nnz / 1e6 * K * 2;
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    printf("nnz: %d, K: %d, time: %f ms\n", nnz, K, milliseconds / PROFILE);
    return (int)(gflop / (milliseconds / PROFILE));
}