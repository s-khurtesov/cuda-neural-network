#pragma once

#include "../common.cuh"
#include "../types/LayerShape.cuh"
#include "../types/Tensor.cuh"

class Layer {
protected:
	std::string name;
	LayerShape shape;

	Tensor x;
	Tensor* y;

public:
	virtual ~Layer() = 0;

	virtual void forward() = 0;
	virtual void backward(float learning_rate, bool front) = 0;

	std::string getName() { return this->name; };
	LayerShape getShape() { return this->shape; };

	Tensor* getX() { return &x; }
	Tensor* getY() { return y; }
	void setX(Tensor& new_y) { x = new_y; }
	void setY(Tensor* new_y) { y = new_y; }

	virtual void init() = 0;
};

inline Layer::~Layer() {}
