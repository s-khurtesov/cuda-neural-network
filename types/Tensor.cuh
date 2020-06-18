#pragma once

#include "../common.cuh"
#include <assert.h>
#include <random>

class Tensor {
private:
	const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
	bool allocated = false;
	bool initialized = false;

public:
	cudnnTensorFormat_t format;
	cudnnTensorDescriptor_t desc;
	float* data;

	int N;
	int C;
	int H;
	int W;

	Tensor()
	{
		format = CUDNN_TENSOR_NCHW;
		desc = NULL;
		data = NULL;
		N = C = H = W = 0;
	}

	Tensor(int N_, int C_, int H_, int W_, cudnnTensorFormat_t format_ = CUDNN_TENSOR_NCHW)
	{
		init(N_, C_, H_, W_, format_);
	}

	void init(int N_, int C_, int H_, int W_, cudnnTensorFormat_t format_ = CUDNN_TENSOR_NCHW)
	{
		N = N_;
		C = C_;
		H = H_;
		W = W_;
		format = format_;
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, format, dataType, N, C, H, W));
		initialized = true;
	}

	void allocate()
	{
		CHECK_CUDA(cudaMallocManaged(&data, size() * sizeof(float)));
		allocated = true;
	}

	int size() { return N * C * H * W; }

	void fill(float val)
	{
		for (int i = 0; i < size(); i++) {
			data[i] = val;
		}
	}

	Tensor& operator=(Tensor& right) {
		if (initialized) {
			assert(right.initialized);
			assert(right.allocated);
			assert(N == right.N);
			assert(C == right.C);
			assert(H == right.H);
			assert(W == right.W);
			assert(format == right.format);
			assert(dataType == right.dataType);
		}
		else {
			init(right.N, right.C, right.H, right.W, right.format);
		}

		if (!allocated)
			allocate();

		CHECK_CUDA(cudaMemcpy(data, right.data, size() * sizeof(float), cudaMemcpyDefault));

		return *this;
	}

	void randomise(float threshold = 1.0f)
	{
		std::default_random_engine generator;
		std::normal_distribution<float> normal_distribution(0.0f, 1.0f);

		for (int i = 0; i < size(); i++) {
			data[i] = normal_distribution(generator) * threshold;
		}
	}

	void show(const char* descr)
	{
		printf("%s:\n", descr);
		for (int n = 0; n < N; n++) {
			printf("    N_%d:\n", n);
			for (int c = 0; c < C; c++) {
				printf("        C_%d:\n", c);
				for (int h = 0; h < H; h++) {
					for (int w = 0; w < W; w++) {
						printf("%11.8f ", data[n * C + c * H + h * W + w]);
					}
					printf("\n");
				}
			}
		}
		putchar('\n');
	}

	~Tensor()
	{
		if (data && allocated) {
			CHECK_CUDA(cudaFree(data));
		}
		if (desc && initialized) {
			CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc));
		}
	}
};
