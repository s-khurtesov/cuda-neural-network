#include "DenseLayer.cuh"

DenseLayer::DenseLayer(
	std::string name_, LayerShape shape_, cublasHandle_t hCublas_)
	: hCublas(hCublas_), ones(NULL)
{
	this->name = name_;
	this->shape = shape_;

	x.init(shape.batch_size, shape.in_nrns, shape.in_nrn_h, shape.in_nrn_w);
	w.init(shape.out_nrns, shape.in_nrns,
		shape.in_nrn_h - shape.out_nrn_h + 1,
		shape.in_nrn_w - shape.out_nrn_w + 1);
	b.init(1, shape.out_nrns, 1, 1);

	dx = x;
	dw = w;
	db = b;

	w.normalDistribution(0.9f);
	dw.fill(0.0f);
	x.fill(0.0f);
	dx.fill(0.0f);
	b.fill(0.0f);
	db.fill(0.0f);

	initOnes();
}

__global__ void FillOnes(float* vec, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	vec[idx] = 1.0f;
}

void DenseLayer::initOnes()
{
	int size = shape.out_nrns * shape.out_nrn_h * shape.out_nrn_w;
	CHECK_CUDA(cudaMallocManaged(&ones, size * sizeof(float)));

	dim3 block_size(128);
	dim3 num_of_blocks((size + block_size.x - 1) / block_size.x);
	FillOnes<<<num_of_blocks, block_size>>>(ones, size);
	CHECK_CUDA(cudaGetLastError());
}

void DenseLayer::init()
{
}

void DenseLayer::forward()
{
	// y = w^T * x
	const int M = shape.out_nrns * shape.out_nrn_h * shape.out_nrn_w;
	const int N = shape.batch_size;
	const int K = shape.in_nrns * shape.in_nrn_h * shape.in_nrn_w;
	CHECK_CUBLAS(cublasSgemm_v2(hCublas, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, alpha, 
		w.data, K, x.data, K, beta, y->data, M));
	// y = y + b * ones
	CHECK_CUBLAS(cublasSgemm_v2(hCublas, CUBLAS_OP_N, CUBLAS_OP_N, M, N, 1, alpha,
		b.data, M, ones, 1, alpha, y->data, M));
}

void DenseLayer::backward(float learning_rate, bool last)
{
	// dw = x * dy^T
	int M = shape.in_nrns * shape.in_nrn_h * shape.in_nrn_w;
	int N = shape.out_nrns * shape.out_nrn_h * shape.out_nrn_w;
	int K = shape.batch_size;
	CHECK_CUBLAS(cublasSgemm_v2(hCublas, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, alpha,
		x.data, M, dy->data, N, beta, dw.data, M));
	// db = dy * ones
	M = shape.out_nrns * shape.out_nrn_h * shape.out_nrn_w;
	N = shape.batch_size;
	CHECK_CUBLAS(cublasSgemv_v2(hCublas, CUBLAS_OP_N, M, N, alpha,
		dy->data, M, ones, 1, beta, db.data, 1));
	if (!last) {
		// dx = w * dy
		M = shape.in_nrns * shape.in_nrn_h * shape.in_nrn_w;
		N = shape.batch_size;
		K = shape.out_nrns * shape.out_nrn_h * shape.out_nrn_w;
		CHECK_CUBLAS(cublasSgemm_v2(hCublas, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alpha,
			w.data, M, dy->data, K, beta, dx.data, M));
	}

	// Update weights and bias
	float learn_alpha = -learning_rate;

	// w = -lr * dw + w
	CHECK_CUBLAS(cublasSaxpy_v2(hCublas, dw.size(), &learn_alpha, dw.data, 1, w.data, 1));
	// b = -lr * db + b
	CHECK_CUBLAS(cublasSaxpy_v2(hCublas, db.size(), &learn_alpha, db.data, 1, b.data, 1));
}

DenseLayer::~DenseLayer()
{
	if (ones) {
		CHECK_CUDA(cudaFree(ones));
	}
}
