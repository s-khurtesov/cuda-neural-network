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
		initialized = true;
	}

	Filter& operator=(Filter& right) {
		if (initialized) {
			assert(right.initialized);
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

		CHECK_CUDA(cudaMemcpy(data, right.data, size() * sizeof(float), cudaMemcpyDefault));

		return *this;
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