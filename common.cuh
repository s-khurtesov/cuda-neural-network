#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <stdio.h>
#include <string.h>
#include <stdexcept>

#define NAN_PROPAGATE		CUDNN_PROPAGATE_NAN
#define NAN_NOT_PROPAGATE	CUDNN_NOT_PROPAGATE_NAN

const char* __stdcall cublasGetErrorString(cublasStatus_t cublasStatus);

void Check(errno_t err, errno_t success, const char* descr, const char* file, const int line);
void CheckCuda(cudaError_t cudaStatus, const char* descr, const char* file, const int line);
void CheckCublas(cublasStatus_t cublasStatus, const char* descr, const char* file, const int line);
void CheckCudnn(cudnnStatus_t cudnnStatus, const char* descr, const char* file, const int line);

errno_t JustCheck(errno_t err, errno_t success, const char* descr, const char* file, const int line);
errno_t JustCheckCuda(cudaError_t cudaStatus, const char* descr, const char* file, const int line);
errno_t JustCheckCublas(cublasStatus_t cublasStatus, const char* descr, const char* file, const int line);
errno_t JustCheckCudnn(cudnnStatus_t cudnnStatus, const char* descr, const char* file, const int line);

#define CHECK(RETERNED, EXPECTED)	Check((RETERNED), (EXPECTED), (#RETERNED), __FILE__, __LINE__)
#define CHECK_CUDA(RETERNED)		CheckCuda((RETERNED), (#RETERNED), __FILE__, __LINE__)
#define CHECK_CUBLAS(RETERNED)		CheckCublas((RETERNED), (#RETERNED), __FILE__, __LINE__)
#define CHECK_CUDNN(RETERNED)		CheckCudnn((RETERNED), (#RETERNED), __FILE__, __LINE__)

#define CHECK_EX(RETERNED, \
			EXPECTED, ON_ERROR)		if (1 == JustCheck((RETERNED), (EXPECTED), (#RETERNED), __FILE__, __LINE__)) \
										{ ON_ERROR; }
#define CHECK_CUDA_EX(RETERNED, \
			ON_ERROR)				if (JustCheckCuda((RETERNED), (#RETERNED), __FILE__, __LINE__)) \
										{ ON_ERROR; }
#define CHECK_CUBLAS_EX(RETERNED, \
			ON_ERROR)				if (JustCheckCublas((RETERNED), (#RETERNED), __FILE__, __LINE__)) \
										{ ON_ERROR; }
#define CHECK_CUDNN_EX(RETERNED, \
			ON_ERROR)				if (JustCheckCudnn((RETERNED), (#RETERNED), __FILE__, __LINE__)) \
										{ ON_ERROR; }

#define printferr(format, ...)		fprintf(stderr, "%s:%-4d "##format"\n", strrchr(__FILE__, '/') + 1, __LINE__, __VA_ARGS__)

#define MAX_COL (6)
#define MAX_LIN (4)

errno_t InitCuda();
errno_t CleanCuda();
errno_t InitCublas(cublasHandle_t*);
errno_t CleanCublas(cublasHandle_t*);
errno_t InitCudnn(cudnnHandle_t*);
errno_t CleanCudnn(cudnnHandle_t*);
