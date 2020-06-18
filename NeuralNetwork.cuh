#pragma once

#include "common.cuh"
#include "types/Tensor.cuh"
#include "layers/Layer.cuh"
#include <vector>

class NeuralNetwork {
private:
	std::vector<Layer*> layers;

public:
	NeuralNetwork();

	void addLayer(Layer*);

	Tensor& forward(Tensor* p_x);
	void backward(Tensor* p_dy, float learning_rate);

	Tensor* getX() { return layers.front()->getX(); }
	Tensor* getY() { return layers.back()->getY(); }
};
