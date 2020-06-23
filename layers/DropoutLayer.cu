#include "DropoutLayer.cuh"

DropoutLayer::DropoutLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_,
	float dropout_, unsigned long long seed_)
	: hCudnn(hCudnn_), dropout(dropout_), seed(seed_), dropoutDesc(NULL), states(NULL),
	statesSizeInBytes(0), reserveSpace(NULL), reserveSpaceSizeInBytes(0)
{
	this->name = name_;
	this->shape = shape_;

	x.init(shape.batch_size, shape.in_nrns, shape.in_nrn_h, shape.in_nrn_w);
	dx = x;

	x.fill(0.0f);
	dx.fill(0.0f);

	CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(x.desc, &reserveSpaceSizeInBytes));
	if (reserveSpaceSizeInBytes) {
		CHECK_CUDA(cudaMallocManaged(&reserveSpace, reserveSpaceSizeInBytes));
	}

	initDropoutDesc();
}

void DropoutLayer::initDropoutDesc()
{
	CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
	CHECK_CUDNN(cudnnDropoutGetStatesSize(hCudnn, &statesSizeInBytes));
	CHECK_CUDA(cudaMallocManaged(&states, statesSizeInBytes));
	CHECK_CUDNN(cudnnSetDropoutDescriptor(dropoutDesc, hCudnn, dropout,
		states, statesSizeInBytes, seed));
}

void DropoutLayer::init()
{
}

void DropoutLayer::forward()
{
	CHECK_CUDNN(cudnnDropoutForward(hCudnn, dropoutDesc, x.desc, x.data, y->desc, y->data, 
		reserveSpace, reserveSpaceSizeInBytes));
}

void DropoutLayer::backward(float learning_rate, bool last)
{
	if (!last) {
		CHECK_CUDNN(cudnnDropoutBackward(hCudnn, dropoutDesc, dy->desc, dy->data, dx.desc, dx.data,
			reserveSpace, reserveSpaceSizeInBytes));
	}
}

DropoutLayer::~DropoutLayer()
{
	if (dropoutDesc) {
		CHECK_CUDNN(cudnnDestroyDropoutDescriptor(dropoutDesc));
	}
	if (reserveSpaceSizeInBytes && reserveSpace) {
		CHECK_CUDA(cudaFree(reserveSpace));
	}
	if (statesSizeInBytes && states) {
		CHECK_CUDA(cudaFree(states));
	}
}
