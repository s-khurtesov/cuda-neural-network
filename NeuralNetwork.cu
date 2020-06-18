#include "NeuralNetwork.cuh"

NeuralNetwork::NeuralNetwork()
{

}

void NeuralNetwork::addLayer(Layer* p_layer)
{
	layers.push_back(p_layer);
}

Tensor& NeuralNetwork::forward(Tensor* p_x)
{
	Tensor* tmp_p_x = p_x;
	for (Layer* cur_layer : layers) {
		tmp_p_x = cur_layer->forward(tmp_p_x);
	}
	return *tmp_p_x;
}

void NeuralNetwork::backward(Tensor* p_dy, float learning_rate)
{
	Tensor* tmp_p_dy = p_dy;
	for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
		tmp_p_dy = (*iter)->backward(tmp_p_dy, learning_rate, *iter == layers.front());
	}
}
