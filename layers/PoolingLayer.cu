#include "PoolingLayer.cuh"

PoolingLayer::PoolingLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_,
	int windowHeight_, int windowWidth_, int verticalStride_, int horizontalStride_,
	int verticalPadding_, int horizontalPadding_, cudnnPoolingMode_t poolingMode_,
	cudnnNanPropagation_t maxpoolingNanOpt_) 
	: hCudnn(hCudnn_), poolingMode(poolingMode_), maxpoolingNanOpt(maxpoolingNanOpt_), 
	windowHeight(windowHeight_), windowWidth(windowWidth_), verticalPadding(verticalPadding_), 
	horizontalPadding(horizontalPadding_), verticalStride(verticalStride_), horizontalStride(horizontalStride_)
{
	this->name = name_;
	this->shape = shape_;

	x.init(shape.batch_size, shape.in_nrns, shape.in_nrn_h, shape.in_nrn_w);
	dx = x;

	x.fill(0.0f);
	dx.fill(0.0f);

	initPoolingDesc();
}

void PoolingLayer::initPoolingDesc()
{
	CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
	CHECK_CUDNN(cudnnSetPooling2dDescriptor(poolingDesc, poolingMode, maxpoolingNanOpt, 
		windowHeight, windowWidth, verticalPadding, horizontalPadding, 
		verticalStride, horizontalStride));
}

void PoolingLayer::init()
{
	int y_n, y_c, y_h, y_w;
	CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(poolingDesc, x.desc, &y_n, &y_c, &y_h, &y_w));
	assert(y->N == y_n);
	assert(y->C == y_c);
	assert(y->H == y_h);
	assert(y->W == y_w);
}

void PoolingLayer::forward()
{
	CHECK_CUDNN(cudnnPoolingForward(hCudnn, poolingDesc, alpha, x.desc, x.data, beta, y->desc, y->data));
}

void PoolingLayer::backward(float learning_rate, bool last)
{
	CHECK_CUDNN(cudnnPoolingBackward(hCudnn, poolingDesc, alpha, y->desc, y->data, dy->desc, dy->data, x.desc, x.data, beta, dx.desc, dx.data));
}

PoolingLayer::~PoolingLayer()
{
	if (poolingDesc) {
		CHECK_CUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
	}
}
