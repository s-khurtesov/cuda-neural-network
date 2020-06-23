#include "ActivationLayer.cuh"

ActivationLayer::ActivationLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_,
	cudnnActivationMode_t activationMode_,
	cudnnNanPropagation_t reluNanOpt_,
	double coef_) : hCudnn(hCudnn_), activationMode(activationMode_), reluNanOpt(reluNanOpt_), 
	coef(coef_), activationDesc(NULL)
{
	this->name = name_;
	this->shape = shape_;

	x.init(shape.batch_size, shape.in_nrns, shape.in_nrn_h, shape.in_nrn_w);
	dx = x;

	x.fill(0.0f);
	dx.fill(0.0f);

	initActivationDesc();
}

void ActivationLayer::initActivationDesc()
{
	CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
	CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, activationMode, reluNanOpt, coef));
}

void ActivationLayer::init() { }

void ActivationLayer::forward()
{
	CHECK_CUDNN(cudnnActivationForward(hCudnn, activationDesc, alpha, x.desc, x.data, beta, y->desc, y->data));
}

void ActivationLayer::backward(float learning_rate, bool last)
{
	CHECK_CUDNN(cudnnActivationBackward(hCudnn, activationDesc, alpha, y->desc, y->data, dy->desc, dy->data, x.desc, x.data, beta, dx.desc, dx.data));
}

ActivationLayer::~ActivationLayer()
{
	if (activationDesc) {
		CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));
	}
}
