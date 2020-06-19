#pragma once

#include "../common.cuh"
#include "Descriptor.cuh"

class Filter : public Descriptor {
public:
	cudnnFilterDescriptor_t desc;

	Filter()
	{
		desc = NULL;
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