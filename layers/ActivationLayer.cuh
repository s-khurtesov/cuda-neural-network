#pragma once

#include "Layer.cuh"
#include "../common.cuh"

#define ACT_SIGMOID			CUDNN_ACTIVATION_SIGMOID
#define ACT_RELU			CUDNN_ACTIVATION_RELU
#define ACT_TANH			CUDNN_ACTIVATION_TANH
#define ACT_CLIPPED_RELU	CUDNN_ACTIVATION_CLIPPED_RELU
#define ACT_ELU				CUDNN_ACTIVATION_ELU

#define NAN_PROPAGATE		CUDNN_PROPAGATE_NAN
#define NAN_NOT_PROPAGATE	CUDNN_NOT_PROPAGATE_NAN

class ActivationLayer : public Layer {
private:
	cudnnHandle_t hCudnn;

	cudnnActivationDescriptor_t activationDesc;
	cudnnActivationMode_t activationMode;
	cudnnNanPropagation_t reluNanOpt;
	double coef;

	void initActivationDesc();

public:
	ActivationLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_,
		cudnnActivationMode_t activationMode = ACT_RELU,
		cudnnNanPropagation_t reluNanOpt = NAN_NOT_PROPAGATE,
		double coef = 1.0f);

	void init();

	void forward();
	void backward(float learning_rate, bool last);

	~ActivationLayer();
};