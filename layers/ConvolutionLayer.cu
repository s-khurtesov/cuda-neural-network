#include "ConvolutionLayer.cuh"

ConvolutionLayer::ConvolutionLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_, 
	cudnnConvolutionFwdAlgo_t algoFwd_, cudnnConvolutionBwdDataAlgo_t algoBwdData_, 
	cudnnConvolutionBwdFilterAlgo_t algoBwdFilter_) 
	: hCudnn(hCudnn_), algoFwd(algoFwd_), algoBwdData(algoBwdData_), algoBwdFilter(algoBwdFilter_)
{
	this->name = name_;
	this->shape = shape_;

	x.init(shape.batch_size, shape.in_nrns, shape.in_nrn_h, shape.in_nrn_w);
	y.init(shape.batch_size, shape.out_nrns, shape.out_nrn_h, shape.out_nrn_w);

	w.init(shape.out_nrns, shape.in_nrns,
		shape.in_nrn_h - shape.out_nrn_h + 1,
		shape.in_nrn_w - shape.out_nrn_w + 1);
	dw.init(shape.out_nrns, shape.in_nrns,
		shape.in_nrn_h - shape.out_nrn_h + 1,
		shape.in_nrn_w - shape.out_nrn_w + 1);
	b.init(1, shape.out_nrns, shape.out_nrn_h, shape.out_nrn_w);
	db.init(1, shape.out_nrns, shape.out_nrn_h, shape.out_nrn_w);

	initConvDesc();

	x.allocate();
	y.allocate();
	b.allocate();
	db.allocate();

	w.randomise();
	dw.fill(0.0f);
	x.fill(0.0f);
	y.fill(0.0f);
	b.fill(0.0f);
	db.fill(0.0f);

	CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
		hCudnn, x.desc, w.desc, convDesc, y.desc, algoFwd, &workSpaceSizeInBytesFwd));
	if (workSpaceSizeInBytesFwd) {
		CHECK_CUDA(cudaMallocManaged(&workSpaceFwd, workSpaceSizeInBytesFwd));
	}
	else {
		workSpaceFwd = NULL;
	}

	CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
		hCudnn, w.desc, y.desc, convDesc, x.desc, algoBwdData, &workSpaceSizeInBytesBwdData));
	if (workSpaceSizeInBytesBwdData) {
		CHECK_CUDA(cudaMallocManaged(&workSpaceBwdData, workSpaceSizeInBytesBwdData));
	}
	else {
		workSpaceBwdData = NULL;
	}

	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		hCudnn, x.desc, y.desc, convDesc, w.desc, algoBwdFilter, &workSpaceSizeInBytesBwdFilter));
	if (workSpaceSizeInBytesBwdFilter) {
		CHECK_CUDA(cudaMallocManaged(&workSpaceBwdFilter, workSpaceSizeInBytesBwdFilter));
	}
	else {
		workSpaceBwdFilter = NULL;
	}
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

Tensor* ConvolutionLayer::forward(Tensor* p_x)
{
	CHECK_CUDNN(cudnnConvolutionForward(
		hCudnn, alpha, p_x->desc, p_x->data, w.desc, w.data,
		convDesc, algoFwd, workSpaceFwd, workSpaceSizeInBytesFwd,
		beta, y.desc, y.data));

	CHECK_CUDNN(cudnnAddTensor(
		hCudnn, alpha, b.desc, b.data, alpha, y.desc, y.data));

	return &y;
}

Tensor* ConvolutionLayer::backward(Tensor* p_dy, float learning_rate, bool last)
{
	CHECK_CUDNN(cudnnConvolutionBackwardBias(
		hCudnn,  alpha,  p_dy->desc, p_dy->data, beta,  db.desc, db.data));

	CHECK_CUDNN(cudnnConvolutionBackwardFilter(
		hCudnn, alpha, x.desc, x.data, p_dy->desc, p_dy->data,
		convDesc, algoBwdFilter,  workSpaceBwdFilter, workSpaceSizeInBytesBwdFilter,
		beta,  dw.desc, dw.data));

	if (!last) {
		CHECK_CUDNN(cudnnConvolutionBackwardData(
			hCudnn,  alpha,  w.desc, w.data,  p_dy->desc, p_dy->data,
			convDesc, algoBwdData,  workSpaceBwdData, workSpaceSizeInBytesBwdData,
			beta,  x.desc, x.data));
	}

	return &x;
}

ConvolutionLayer::~ConvolutionLayer()
{
	if (workSpaceFwd) {
		cudaFree(workSpaceFwd);
	}
	if (workSpaceBwdFilter) {
		cudaFree(workSpaceBwdFilter);
	}
	if (workSpaceBwdData) {
		cudaFree(workSpaceBwdData);
	}
	if (convDesc) {
		cudnnDestroyConvolutionDescriptor(convDesc);
	}
}
