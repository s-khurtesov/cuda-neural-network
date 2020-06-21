#include "NeuralNetwork.cuh"

extern __device__ float atomicAdd(float* address, float val);

void NeuralNetwork::init()
{
	assert(!initialized);
	assert(layers.size() >= 1);

	Tensor* tmp_x = layers.front()->getX();
	LayerShape last_shape = layers.back()->getShape();

	x.init(tmp_x->N, tmp_x->C, tmp_x->H, tmp_x->W, tmp_x->format);
	y.init(last_shape.batch_size, last_shape.out_nrns, last_shape.out_nrn_h, last_shape.out_nrn_w);
	dy = y;

	x.fill(0);
	y.fill(0);
	dy.fill(0);

	for (auto iter = layers.begin(); iter != layers.end() - 1; iter++) {
		(*iter)->setY((*(iter + 1))->getX());
		(*iter)->setdY((*(iter + 1))->getdX());
	}
	layers.back()->setY(&y);
	layers.back()->setdY(&dy);

	for (Layer* cur_layer : layers) {
		cur_layer->init();
	}

	initialized = true;
}

void NeuralNetwork::clear()
{
	assert(initialized);

	layers.clear();

	initialized = false;
}

void NeuralNetwork::addLayer(Layer* p_layer)
{
	assert(!initialized);

	layers.push_back(p_layer);
}

Tensor& NeuralNetwork::forward(Tensor& x)
{
	assert(initialized);

	layers.front()->setX(x);
	for (Layer* cur_layer : layers) {
		cur_layer->forward();
	}
	return y;
}

void NeuralNetwork::backward(Tensor& dy, float learning_rate)
{
	assert(initialized);

	this->dy = dy;
	for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
		(*iter)->backward(learning_rate, *iter == layers.front());
	}
}

__global__ void binaryCrossEntropyCost(float* predictions, float* target, int size, float* cost) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size && fabsf(target[index] - predictions[index]) > 0.0000001f) {
		float partial_cost = target[index] * logf(predictions[index])
			+ (1.0f - target[index]) * logf(1.0f - predictions[index]);
		atomicAdd(cost, -partial_cost / size);
	}
}

__global__ void dBinaryCrossEntropyError(float* predictions, float* target, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size && fabsf(target[index] - predictions[index]) > 0.0000001f) {
		predictions[index] = -1.0 * (target[index] / predictions[index] - (1 - target[index]) / (1 - predictions[index]));
	}
}

void NeuralNetwork::calcError(Tensor& labels)
{
	dim3 block_size(128);
	dim3 num_of_blocks((dy.size() + block_size.x - 1) / block_size.x);
	dy = y;
	dBinaryCrossEntropyError<<<num_of_blocks, block_size>>>(
		dy.data,
		labels.data,
		dy.size());
	CHECK_CUDA(cudaGetLastError());
}

void NeuralNetwork::calcCost(Tensor& labels, float* cost)
{
	dim3 block_size(128);
	dim3 num_of_blocks((y.size() + block_size.x - 1) / block_size.x);
	binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(
		y.data,
		labels.data,
		y.size(), cost);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());
}

void NeuralNetwork::train(Tensor& x, Tensor& labels, int iters, float learning_rate, float learning_rate_lowering_coef)
{
	assert(initialized);
	assert(y.N == labels.N);
	assert(y.C == labels.C);
	assert(y.H == labels.H);
	assert(y.W == labels.W);
	assert(y.format == labels.format);

	float cur_learning_rate = learning_rate;
	float* cost;
	int period = iters / 10;
	int short_period = iters / 100;
	float lr_decrement = (1.0f - learning_rate_lowering_coef) * learning_rate / iters;

	if (!period)
		period = 1;
	if (!short_period)
		short_period = 1;

	CHECK_CUDA(cudaMallocManaged(&cost, sizeof(float)));

	layers.front()->setX(x);

	CHECK_CUDA(cudaDeviceSynchronize());

	for (int iteration = 0; iteration < iters; iteration++) {
		*cost = 0.0f;
		cur_learning_rate = cur_learning_rate - lr_decrement;

		for (Layer* cur_layer : layers) {
			cur_layer->forward();
		}

		calcError(labels);

		for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
			(*iter)->backward(learning_rate, *iter == layers.front());
		}

		calcCost(labels, cost);

		if ((iteration < period && (iteration + 1) % (short_period) == 0) || (iteration + 1) % (period) == 0) {
			printf("Iteration: %d, Cost: %f, learning_rate: %f\n", iteration + 1, *cost, cur_learning_rate);
		}
	}

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaFree(cost));
}
