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

void Check(errno_t err, errno_t success, const char* descr, const char* file, const int line)
{
    DWORD dErr = GetLastError();
    if (err != success) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d (returned %d): %s", strrchr(file, '/') + 1, line, descr, dErr, err, GetErrorString(dErr).c_str());
        CleanCuda();
        exit(EXIT_FAILURE);
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
        exit(EXIT_FAILURE);
    }
}

void CheckCudnn(cudnnStatus_t cudnnStatus, const char* descr, const char* file, const int line)
{
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "%s:%-4d %s\nERROR #%d: %s", strrchr(file, '/') + 1, line, descr, cudnnStatus, cudnnGetErrorString(cudnnStatus));
        CleanCuda();
        exit(EXIT_FAILURE);
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

errno_t InitCudnn(cudnnHandle_t* handle)
{
    CHECK_CUDNN(cudnnCreate(handle));

    return 0;
}

errno_t CleanCudnn(cudnnHandle_t* handle)
{
    CHECK_CUDNN(cudnnDestroy(*handle));

    return 0;
}
