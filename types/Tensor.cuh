#pragma once

#include "../common.cuh"
#include "Descriptor.cuh"

class Tensor : public Descriptor {
private:
	bool allocated = false;
	bool initialized = false;

public:
	cudnnTensorDescriptor_t desc;

	Tensor()
	{
		desc = NULL;
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
		assert(initialized);

		CHECK_CUDA(cudaMallocManaged(&data, size() * sizeof(float)));
		allocated = true;
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
