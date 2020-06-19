#include "common.cuh"

#include <stdlib.h>
#include <windows.h>
#include <string>

std::string GetErrorString(DWORD errorMessageID)
{
    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, errorMessageID, MAKELANGID(LANG_ENGLISH, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

    std::string message(messageBuffer, size);

    //Free the buffer.
    LocalFree(messageBuffer);

    return message;
}

const char* __stdcall cublasGetErrorString(cublasStatus_t cublasStatus)
{
    switch (cublasStatus) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "Undefined cuBLAS status";
    }
}

void Check(errno_t err, errno_t success, const char* descr, const char* file, const int line)
{
    DWORD dErr = GetLastError();
    if (err != success) {
        std::string msg = GetErrorString(dErr);
        fprintf(stderr, "%s:%-4d %s\nERROR #%d (returned %d): %s", strrchr(file, '/') + 1, line, descr, dErr, err, msg.c_str());
        CleanCuda();
        throw std::runtime_error(msg.c_str());
    }
    else if (dErr != 0) {
        fprintf(stderr, "%s:%-4d WARNING: Returned as expected, but error #%d (returned %d): %s\n", strrchr(__FILE__, '/') + 1, __LINE__, dErr, err, GetErrorString(dErr).c_str());
    }
}

void CheckCuda(cudaError_t cudaStatus, const char* descr, const char* file, const int line)
{
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d: %s", strrchr(file, '/') + 1, line, descr, cudaStatus, cudaGetErrorString(cudaStatus));
        CleanCuda();
        throw std::runtime_error(cudaGetErrorString(cudaStatus));
    }
}

void CheckCublas(cublasStatus_t cublasStatus, const char* descr, const char* file, const int line)
{
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d: %s", strrchr(file, '/') + 1, line, descr, cublasStatus, cublasGetErrorString(cublasStatus));
        CleanCuda();
        throw std::runtime_error(cublasGetErrorString(cublasStatus));
    }
}

void CheckCudnn(cudnnStatus_t cudnnStatus, const char* descr, const char* file, const int line)
{
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d: %s", strrchr(file, '/') + 1, line, descr, cudnnStatus, cudnnGetErrorString(cudnnStatus));
        CleanCuda();
        throw std::runtime_error(cudnnGetErrorString(cudnnStatus));
    }
}

errno_t JustCheck(errno_t err, errno_t success, const char* descr, const char* file, const int line)
{
    DWORD dErr = GetLastError();
    if (err != success) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d (returned %d): %s", strrchr(file, '/') + 1, line, descr, dErr, err, GetErrorString(dErr).c_str());
        return 1;
    }
    else if (dErr != 0) {
        fprintf(stderr, "%s:%-4d WARNING: Returned as expected, but error #%d (returned %d): %s\n", strrchr(__FILE__, '/') + 1, __LINE__, dErr, err, GetErrorString(dErr).c_str());
        return 2;
    }
    return 0;
}

errno_t JustCheckCuda(cudaError_t cudaStatus, const char* descr, const char* file, const int line)
{
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d: %s", strrchr(file, '/') + 1, line, descr, cudaStatus, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

errno_t JustCheckCublas(cublasStatus_t cublasStatus, const char* descr, const char* file, const int line)
{
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d: %s", strrchr(file, '/') + 1, line, descr, cublasStatus, cublasGetErrorString(cublasStatus));
        return 1;
    }
    return 0;
}

errno_t JustCheckCudnn(cudnnStatus_t cudnnStatus, const char* descr, const char* file, const int line)
{
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d: %s", strrchr(file, '/') + 1, line, descr, cudnnStatus, cudnnGetErrorString(cudnnStatus));
        return 1;
    }
    return 0;
}

errno_t InitCuda()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    CHECK_CUDA(cudaSetDevice(0));

    return 0;
}

errno_t CleanCuda()
{
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    CHECK_CUDA_EX(cudaDeviceReset(), return 1);

    return 0;
}

errno_t InitCublas(cublasHandle_t* handleCublas)
{
    cublasCreate_v2(handleCublas);

    return 0;
}

errno_t CleanCublas(cublasHandle_t* handleCublas)
{

    return 0;
}

errno_t InitCudnn(cudnnHandle_t* handleCudnn)
{
    CHECK_CUDNN(cudnnCreate(handleCudnn));

    return 0;
}

errno_t CleanCudnn(cudnnHandle_t* handleCudnn)
{
    CHECK_CUDNN(cudnnDestroy(*handleCudnn));

    return 0;
}
