#pragma once

#include "common.cuh"
#include "types/Tensor.cuh"
#include "layers/Layer.cuh"
#include <vector>

class NeuralNetwork {
private:
	bool initialized = false;
	std::vector<Layer*> layers;
	Tensor x;
	Tensor y;
	Tensor dy;

	void clampOutput(float min = 0.0f + FLT_EPSILON, float max = 1.0f - FLT_EPSILON);
	void calcError(Tensor& labels);
	void calcCost(Tensor& labels, float* cost);

public:
	NeuralNetwork() { }

	void init();
	void clear();

	void addLayer(Layer*);

	Tensor& forward(Tensor& x);
	void backward(Tensor& dy, float learning_rate);

	void train(Tensor& x, Tensor& labels, int iters, float learning_rate, float learning_rate_lowering_coef = 1.0f);

	Tensor& getX() { return x; }
	Tensor& getY() { return y; }
};
