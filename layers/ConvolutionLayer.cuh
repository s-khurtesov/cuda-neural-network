#pragma once

#include "../common.cuh"
#include "../types/LayerShape.cuh"
#include "../types/Filter.cuh"
#include "../types/Tensor.cuh"
#include "Layer.cuh"

class ConvolutionLayer : public Layer {
private:
	cudnnHandle_t hCudnn;
	cublasHandle_t hCublas;

	Filter w;
	Filter dw;
	Tensor b;
	Tensor db;

	float* workSpaceFwd;
	size_t workSpaceSizeInBytesFwd;

	float* workSpaceBwdData;
	size_t workSpaceSizeInBytesBwdData;

	float* workSpaceBwdFilter;
	size_t workSpaceSizeInBytesBwdFilter;

	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t algoFwd;

	cudnnConvolutionBwdDataAlgo_t algoBwdData;
	cudnnConvolutionBwdFilterAlgo_t algoBwdFilter;

	void initConvDesc(int pad_h = 0, int pad_w = 0, 
		int stride_h = 1, int stride_w = 1, int dil_h = 1, int dil_w = 1, 
		cudnnConvolutionMode_t convDescMode = CUDNN_CONVOLUTION,
		cudnnDataType_t convDescComputeType = CUDNN_DATA_FLOAT);

public:
	ConvolutionLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_, cublasHandle_t hCublas_, 
		cudnnConvolutionFwdAlgo_t algoFwd = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
		cudnnConvolutionBwdDataAlgo_t algoBwdData = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
		cudnnConvolutionBwdFilterAlgo_t algoBwdFilter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);

	void init();

	void forward();
	void backward(float learning_rate, bool last);

	~ConvolutionLayer();
};
