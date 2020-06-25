#pragma once

#include "../common.cuh"
#include "../types/LayerShape.cuh"
#include "../types/Filter.cuh"
#include "../types/Tensor.cuh"
#include "Layer.cuh"

class DenseLayer : public Layer {
private:
	cublasHandle_t hCublas;

	Filter w;
	Filter dw;
	Tensor b;
	Tensor db;

	float* ones;

	void initOnes();

public:
	DenseLayer(std::string name_, LayerShape shape_, cublasHandle_t hCublas_, float filterScale = 0.0f);

	void init();

	void forward();
	void backward(float learning_rate, bool last);

	~DenseLayer();
};
