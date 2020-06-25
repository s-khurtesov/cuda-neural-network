#include "ConvolutionLayer.cuh"

ConvolutionLayer::ConvolutionLayer(
	std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_, cublasHandle_t hCublas_, float filterScale,
	int dil_h, int dil_w, int stride_h, int stride_w, int pad_h, int pad_w,
	cudnnConvolutionFwdAlgo_t algoFwd_, cudnnConvolutionBwdDataAlgo_t algoBwdData_,
	cudnnConvolutionBwdFilterAlgo_t algoBwdFilter_)
	: hCudnn(hCudnn_), hCublas(hCublas_), algoFwd(algoFwd_), algoBwdData(algoBwdData_), algoBwdFilter(algoBwdFilter_), 
	workSpaceFwd(NULL), workSpaceSizeInBytesFwd(0), workSpaceBwdData(NULL), workSpaceSizeInBytesBwdData(0), 
	workSpaceBwdFilter(NULL), workSpaceSizeInBytesBwdFilter(0), convDesc(NULL)
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

	if (filterScale == 0.0f) {
		w.normalDistribution(1.0f / sqrtf((float)(w.C * w.H * w.W)));
	}
	else {
		w.normalDistribution(filterScale);
	}
	dw.fill(0.0f);
	x.fill(0.0f);
	dx.fill(0.0f);
	b.fill(0.0f);
	db.fill(0.0f);

	initConvDesc(pad_h, pad_w, stride_h, stride_w, dil_h, dil_w);
}

void ConvolutionLayer::initConvDesc(int pad_h, int pad_w,
	int stride_h, int stride_w, int dil_h, int dil_w,
	cudnnConvolutionMode_t convDescMode, cudnnDataType_t convDescComputeType)
{
	CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
		convDesc, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w,
		convDescMode, convDescComputeType));
}

void ConvolutionLayer::init()
{
	int y_n, y_c, y_h, y_w;
	CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, x.desc, w.desc, &y_n, &y_c, &y_h, &y_w));
	assert(y->N == y_n);
	assert(y->C == y_c);
	assert(y->H == y_h);
	assert(y->W == y_w);

	CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
		hCudnn, x.desc, w.desc, convDesc, y->desc, algoFwd, &workSpaceSizeInBytesFwd));
	if (workSpaceSizeInBytesFwd) {
		CHECK_CUDA(cudaMallocManaged(&workSpaceFwd, workSpaceSizeInBytesFwd));
	}
	else {
		workSpaceFwd = NULL;
	}

	CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
		hCudnn, w.desc, y->desc, convDesc, x.desc, algoBwdData, &workSpaceSizeInBytesBwdData));
	if (workSpaceSizeInBytesBwdData) {
		CHECK_CUDA(cudaMallocManaged(&workSpaceBwdData, workSpaceSizeInBytesBwdData));
	}
	else {
		workSpaceBwdData = NULL;
	}

	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		hCudnn, x.desc, y->desc, convDesc, w.desc, algoBwdFilter, &workSpaceSizeInBytesBwdFilter));
	if (workSpaceSizeInBytesBwdFilter) {
		CHECK_CUDA(cudaMallocManaged(&workSpaceBwdFilter, workSpaceSizeInBytesBwdFilter));
	}
	else {
		workSpaceBwdFilter = NULL;
	}
}

void ConvolutionLayer::forward()
{
	CHECK_CUDNN(cudnnConvolutionForward(
		hCudnn, alpha, x.desc, x.data, w.desc, w.data,
		convDesc, algoFwd, workSpaceFwd, workSpaceSizeInBytesFwd,
		beta, y->desc, y->data));

	CHECK_CUDNN(cudnnAddTensor(// ??????????????????????????????????????????????????????????????????
		hCudnn, alpha, b.desc, b.data, alpha, y->desc, y->data));
}

void ConvolutionLayer::backward(float learning_rate, bool last)
{
	CHECK_CUDNN(cudnnConvolutionBackwardBias(
		hCudnn, alpha, dy->desc, dy->data, beta, db.desc, db.data));

	CHECK_CUDNN(cudnnConvolutionBackwardFilter(
		hCudnn, alpha, x.desc, x.data, dy->desc, dy->data,
		convDesc, algoBwdFilter, workSpaceBwdFilter, workSpaceSizeInBytesBwdFilter,
		beta, dw.desc, dw.data));

	if (!last) {
		CHECK_CUDNN(cudnnConvolutionBackwardData(
			hCudnn, alpha, w.desc, w.data, dy->desc, dy->data,
			convDesc, algoBwdData, workSpaceBwdData, workSpaceSizeInBytesBwdData,
			beta, dx.desc, dx.data));
	}

	// Update Filter and Bias
	float learn_alpha = -learning_rate;
	CHECK_CUBLAS(cublasSaxpy_v2(hCublas, w.size(), &learn_alpha, dw.data, 1, w.data, 1));
	CHECK_CUBLAS(cublasSaxpy_v2(hCublas, b.size(), &learn_alpha, db.data, 1, b.data, 1));
}

ConvolutionLayer::~ConvolutionLayer()
{
	if (workSpaceFwd) {
		CHECK_CUDA(cudaFree(workSpaceFwd));
	}
	if (workSpaceBwdFilter) {
		CHECK_CUDA(cudaFree(workSpaceBwdFilter));
	}
	if (workSpaceBwdData) {
		CHECK_CUDA(cudaFree(workSpaceBwdData));
	}
	if (convDesc) {
		CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	}
}
