#pragma once

#include "Layer.cuh"
#include "../common.cuh"

class DropoutLayer : public Layer {
private:
	cudnnHandle_t hCudnn;

	cudnnDropoutDescriptor_t dropoutDesc = NULL;
	float dropout;
	float* states = NULL;
	size_t statesSizeInBytes = 0;
	unsigned long long seed;
	float* reserveSpace = NULL;
	size_t reserveSpaceSizeInBytes = 0;

	void initDropoutDesc();

public:
	DropoutLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_,
		float dropout_ = 0.1, unsigned long long seed_ = 0);

	void init();

	void forward();
	void backward(float learning_rate, bool last);

	~DropoutLayer();
};
