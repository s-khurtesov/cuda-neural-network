#pragma once

#include "../common.cuh"
#include <random>

class Filter {
private:
	const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t format;

public:
	cudnnFilterDescriptor_t desc;
	float* data;

	int N;
	int C;
	int H;
	int W;

	Filter()
	{
		format = CUDNN_TENSOR_NCHW;
		desc = NULL;
		data = NULL;
		N = C = H = W = 0;
	}

	Filter(int N_, int C_, int H_, int W_, cudnnTensorFormat_t format_ = CUDNN_TENSOR_NCHW)
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
		CHECK_CUDNN(cudnnCreateFilterDescriptor(&desc));
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(desc, dataType, format, N, C, H, W));
		CHECK_CUDA(cudaMallocManaged(&data, size() * sizeof(float)));
	}

	int size() { return N * C * H * W; }

	void randomise(float threshold = 1.0f)
	{
		std::default_random_engine generator;
		std::normal_distribution<float> normal_distribution(0.0f, 1.0f);

		for (int i = 0; i < size(); i++) {
			data[i] = normal_distribution(generator) * threshold;
		}
	}

	void fill(float val)
	{
		for (int i = 0; i < size(); i++) {
			data[i] = val;
		}
	}

	~Filter()
	{
		if (data) {
			CHECK_CUDA(cudaFree(data));
		}
		if (desc) {
			CHECK_CUDNN(cudnnDestroyFilterDescriptor(desc));
		}
	}
};