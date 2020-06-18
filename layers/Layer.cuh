#pragma once

#include "../common.cuh"
#include "../types/LayerShape.cuh"
#include "../types/Tensor.cuh"

class Layer {
protected:
	std::string name;
	LayerShape shape;

public:
	virtual ~Layer() = 0;

	virtual Tensor* forward(Tensor* p_x) = 0;
	virtual Tensor* backward(Tensor* p_dy, float learning_rate, bool front) = 0;

	std::string getName() { return this->name; };

	virtual Tensor* getX() = 0;
	virtual Tensor* getY() = 0;
};

inline Layer::~Layer() {}
