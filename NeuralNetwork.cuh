#pragma once

#include "common.cuh"
#include "types/Tensor.cuh"
#include "types/ImageDataset.cuh"
#include "layers/Layer.cuh"
#include <vector>

class NeuralNetwork {
private:
	bool initialized = false;
	std::vector<Layer*> layers;
	Tensor y;
	Tensor dy;

	void clampOutput(float min = 0.0f + FLT_EPSILON, float max = 1.0f - FLT_EPSILON);
	void calcError(Tensor& labels);
	void calcCost(Tensor& labels, float* cost);
	void calcAccuracy(Tensor& y, Tensor& targets, int* p_right_ones, int* p_right_zeros, int* p_all_ones);

public:
	NeuralNetwork() { }

	void init();
	void clear();

	void addLayer(Layer*);

	Tensor& forward(Tensor& x);
	void backward(Tensor& dy, float learning_rate);

	void train(ImageDataset& dataset, int epochs, float learning_rate, float learning_rate_lowering_coef = 1.0f, float earlyStop = 0.1f, bool debug = false);

	void summary();

	Tensor& getX() { return *layers.front()->getX(); }
	Tensor& getY() { return y; }
};
