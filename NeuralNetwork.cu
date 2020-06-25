#include "NeuralNetwork.cuh"

extern __device__ float atomicAdd(float* address, float val);

void NeuralNetwork::init()
{
	assert(!initialized);
	assert(layers.size() >= 1);

	Tensor* tmp_x = layers.front()->getX();
	LayerShape last_shape = layers.back()->getShape();

	y.init(last_shape.batch_size, last_shape.out_nrns, last_shape.out_nrn_h, last_shape.out_nrn_w);
	dy = y;

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
	CHECK_CUDA(cudaDeviceSynchronize());
	return y;
}

void NeuralNetwork::backward(Tensor& dy, float learning_rate)
{
	assert(initialized);

	this->dy = dy;
	for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
		(*iter)->backward(learning_rate, *iter == layers.front());
	}
	CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void binaryCrossEntropyCost(float* predictions, float* target, int size, float* cost) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		float partial_cost = target[index] * logf(predictions[index])
			+ (1.0f - target[index]) * logf(1.0f - predictions[index]);
		atomicAdd(cost, -partial_cost / size);
	}
}

__global__ void dBinaryCrossEntropyError(float* predictions, float* target, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		predictions[index] = -1.0f * (target[index] / predictions[index]
			- (1.0f - target[index]) / (1.0f - predictions[index]));
	}
}

__global__ void clamp(float* predictions, int size, float min, float max) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		predictions[index] = fmaxf(min, fminf(predictions[index], max));
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

void NeuralNetwork::clampOutput(float min, float max)
{
	dim3 block_size(128);
	dim3 num_of_blocks((y.size() + block_size.x - 1) / block_size.x);
	clamp<<<num_of_blocks, block_size>>>(
		y.data, y.size(), min, max);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());
}

void NeuralNetwork::train(ImageDataset& dataset, int epochs, float learning_rate, float learning_rate_lowering_coef, float earlyStop, bool debug)
{
	assert(initialized);

	float cur_learning_rate = learning_rate;
	float* cost;
	int period = dataset.size() / 5;
	int short_period = period / 5;
	float lr_decrement = epochs > 1 ? (1.0f - learning_rate_lowering_coef) * learning_rate / (epochs - 1) : 0.0f;

	if (!period)
		period = 1;
	if (!short_period)
		short_period = 1;

	CHECK_CUDA(cudaMallocManaged(&cost, sizeof(float)));

	CHECK_CUDA(cudaDeviceSynchronize());

	auto start = std::chrono::high_resolution_clock::now();
	for (int epoch = 0; epoch < epochs; epoch++) {
		// Init epoch
		int right_ones = 0, right_zeros = 0, all_ones = 0, all_zeros = 0;
		float lr_batch_increment = dataset.size() > 1 ? (1.0f - learning_rate_lowering_coef) * cur_learning_rate / (dataset.size() - 1) : 0.0f;
		float batch_learning_rate;

		if (!(epoch & 1)) {
			batch_learning_rate = cur_learning_rate * learning_rate_lowering_coef;
		}
		else {
			batch_learning_rate = cur_learning_rate;
			lr_batch_increment = -lr_batch_increment;
		}

		float epoch_cost = 0.0f;

		for (int batch = 0; batch < dataset.size(); batch++) {
			*cost = 0.0f;
			Tensor& input = dataset.getInput(batch);
			Tensor& targets = dataset.getTarget(batch);

			// TODO: Make this more efficient
			layers.front()->setX(input);

			// Forward Propagation
			for (Layer* cur_layer : layers) {
				cur_layer->forward();
			}

			// Clamp output to avoid infinity in Binary Cross Entropy calculations
			clampOutput();

			// Calc dy
			calcError(targets);

			// Backpropagtion
			for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
				(*iter)->backward(batch_learning_rate, *iter == layers.front());
			}

			// Calc statistics and print it sometimes
			calcCost(targets, cost);
			epoch_cost += *cost;
			calcAccuracy(y, targets, &right_ones, &right_zeros, &all_ones, &all_zeros);
			if ((batch < period && (batch + 1) % (short_period) == 0) || (batch + 1) % (period) == 0) {
				if (debug) {
					printf("    Batch: %4d, Cost: %1.4f, learning_rate: %1.5f, y_int: %4d, dy_nan: %4d, dy_inf: %4d", batch + 1, *cost, batch_learning_rate,
						std::count_if(y.data, y.data + y.size() - 1, [](float x) {return (int)x == (float)x; }),
						std::count_if(dy.data, dy.data + dy.size() - 1, [](float x) {return isnan(x); }),
						std::count_if(dy.data, dy.data + dy.size() - 1, [](float x) {return isinf(x); }));
				}
				else {
					printf("    Batch: %4d, Cost: %1.4f, learning_rate: %1.5f", batch + 1, *cost, batch_learning_rate);
				}

				printf(", Accuracy: %1.2f, Ones: %1.2f (%4d/%4d), Zeros: %1.2f (%4d/%4d)\n", (float)(right_ones + right_zeros) / (all_ones + all_zeros),
					all_ones ? (float)(right_ones) / all_ones : 0.0f, right_ones, all_ones, all_zeros ? (float)(right_zeros) / all_zeros : 0.0f, right_zeros, all_zeros);
				right_ones = right_zeros = all_ones = all_zeros = 0;
			}

			// Increment learning rate for each batch
			batch_learning_rate += lr_batch_increment;
		}
		// Print epoch summary
		epoch_cost /= dataset.size();
		printf("Epoch: %3d, Cost: %1.4f, learning_rate: %1.5f-%1.5f\n\n", epoch + 1, epoch_cost, 
			cur_learning_rate - (!(epoch & 1) ? lr_batch_increment * dataset.size() : 0.0f), 
			cur_learning_rate - ((epoch & 1) ? -lr_batch_increment * dataset.size() : 0.0f));

		// Early stop
		if (epoch_cost <= earlyStop) {
			break;
		}

		// Decrement learning rate for each epoch
		cur_learning_rate -= lr_decrement;
	}

	CHECK_CUDA(cudaDeviceSynchronize());

	// Print elapsed time
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	std::cout << "Time: " << duration.count() << 's' << std::endl;

	CHECK_CUDA(cudaFree(cost));
}

void NeuralNetwork::calcAccuracy(Tensor& y, Tensor& targets, int* p_right_ones, int* p_right_zeros, int* p_all_ones, int* p_all_zeros)
{
	int right_ones = 0;
	int right_zeros = 0;
	int all_ones = 0;

	for (int n = 0; n < targets.N; n++) {
		float prediction = (y.data[n] > 0.5f) ? 1.0f : 0.0f;
		if (prediction == targets.data[n]) {
			if (prediction == 1.0f) {
				right_ones++;
			}
			else {
				right_zeros++;
			}
		}
		all_ones += prediction;
	}
	*p_right_ones += right_ones;
	*p_right_zeros += right_zeros;
	*p_all_ones += all_ones;
	*p_all_zeros += targets.N - all_ones;
}

void NeuralNetwork::summary()
{
	printf("Neural Network Summary:\n");
	for (Layer* layer : layers) {
		printf("%s:\n In:\t%d x %d x %d\n Out:\t%d x %d x %d\n\n", layer->getName().c_str(),
			layer->getShape().in_nrns, layer->getShape().in_nrn_h, layer->getShape().in_nrn_w,
			layer->getShape().out_nrns, layer->getShape().out_nrn_h, layer->getShape().out_nrn_w);
	}
}
