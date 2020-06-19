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

	x.allocate();
	y.allocate();

	x.fill(0);
	y.fill(0);

	for (auto iter = layers.begin(); iter != layers.end() - 1; iter++) {
		(*iter)->setY((*(iter + 1))->getX());
	}
	layers.back()->setY(&y);

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

	y = dy;
	for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
		(*iter)->backward(learning_rate, *iter == layers.front());
	}
}

__global__ void binaryCrossEntropyCost(float* predictions, float* target,
	int size, float* cost) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		float partial_cost = target[index] * logf(predictions[index])
			+ (1.0f - target[index]) * logf(1.0f - predictions[index]);
		atomicAdd(cost, -partial_cost / size);
	}
}

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target,
	int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		predictions[index] = -1.0 * (target[index] / predictions[index] - (1 - target[index]) / (1 - predictions[index]));
	}
}

void NeuralNetwork::calcError(Tensor& labels)
{
	dim3 block_size(128);
	dim3 num_of_blocks((y.size() + block_size.x - 1) / block_size.x);
	dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(
		y.data,
		labels.data,
		y.size());
	CHECK_CUDA(cudaGetLastError());
}

void NeuralNetwork::calcCost(Tensor& labels, float* cost)
{
	dim3 block_size(124);
	dim3 num_of_blocks((y.size() + block_size.x - 1) / block_size.x);
	binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(
		y.data,
		labels.data,
		y.size(), cost);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());
}

void NeuralNetwork::train(Tensor& x, Tensor& labels, int iters, float learning_rate)
{
	assert(initialized);
	assert(y.N == labels.N);
	assert(y.C == labels.C);
	assert(y.H == labels.H);
	assert(y.W == labels.W);
	assert(y.format == labels.format);

	float* iterCost;
	cudaMallocManaged(&iterCost, sizeof(float));
	int period = iters / 10;
	if (!period)
		period = 1;

	layers.front()->setX(x);

	CHECK_CUDA(cudaDeviceSynchronize());

	for (int iteration = 0; iteration < iters; iteration++) {
		float cost = 0.0f;
		*iterCost = 0.0f;

		for (Layer* cur_layer : layers) {
			cur_layer->forward();
		}

		calcError(labels);

		for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
			(*iter)->backward(learning_rate, *iter == layers.front());
		}

		calcCost(labels, iterCost);
		cost += *iterCost;

		if ((iteration + 1) % (period) == 0) {
			printf("Iteration: %d, Cost: %f\n", iteration + 1, cost / 1/*number of batches*/);
		}
	}

	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaFree(iterCost));
}
