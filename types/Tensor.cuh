#pragma once

#include "../common.cuh"
#include <assert.h>
#include <random>

class Tensor {
private:
	const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t format;
	bool allocated = false;
	bool initialized = false;

public:
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
		assert(initialized);
		assert(right.initialized);
		assert(right.allocated);
		assert(N == right.N);
		assert(C == right.C);
		assert(H == right.H);
		assert(W == right.W);
		assert(format == right.format);
		assert(dataType == right.dataType);

		if (!allocated)
			allocate();

		CHECK_CUDA(cudaMemcpy(data, right.data, size() * sizeof(float), cudaMemcpyDefault));

		return right;
	}

	void randomise(float threshold = 1.0f)
	{
		std::default_random_engine generator;
		std::normal_distribution<float> normal_distribution(0.0f, 1.0f);

		for (int i = 0; i < size(); i++) {
			data[i] = normal_distribution(generator) * threshold;
		}
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
