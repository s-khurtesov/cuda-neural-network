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

	void initConvDesc(int pad_h, int pad_w, 
		int stride_h, int stride_w, int dil_h, int dil_w, 
		cudnnConvolutionMode_t convDescMode = CUDNN_CONVOLUTION,
		cudnnDataType_t convDescComputeType = CUDNN_DATA_FLOAT);

public:
	ConvolutionLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_, cublasHandle_t hCublas_, float filterScale = 0.0f, 
		int dil_h = 3, int dil_w = 3, int stride_h = 1, int stride_w = 1, int pad_h = 2, int pad_w = 2,
		cudnnConvolutionFwdAlgo_t algoFwd = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
		cudnnConvolutionBwdDataAlgo_t algoBwdData = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
		cudnnConvolutionBwdFilterAlgo_t algoBwdFilter = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);

	void init();

	void forward();
	void backward(float learning_rate, bool last);

	~ConvolutionLayer();
};
