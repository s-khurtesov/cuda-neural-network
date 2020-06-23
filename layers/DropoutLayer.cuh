#pragma once

#include "Layer.cuh"
#include "../common.cuh"

class DropoutLayer : public Layer {
private:
	cudnnHandle_t hCudnn;

	cudnnDropoutDescriptor_t dropoutDesc;
	float dropout;
	float* states;
	size_t statesSizeInBytes;
	unsigned long long seed;
	float* reserveSpace;
	size_t reserveSpaceSizeInBytes;

	void initDropoutDesc();

public:
	DropoutLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_,
		float dropout_ = 0.1, unsigned long long seed_ = 0);

	void init();

	void forward();
	void backward(float learning_rate, bool last);

	~DropoutLayer();
};
