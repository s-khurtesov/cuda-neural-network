#pragma once

#include "Layer.cuh"
#include "../common.cuh"

#define POOL_MAX				CUDNN_POOLING_MAX
#define POOL_AVG_WITH_PADDING	CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
#define POOL_AVG				CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
#define POOL_MAX_DET			CUDNN_POOLING_MAX_DETERMINISTIC

class PoolingLayer : public Layer {
private:
	cudnnHandle_t hCudnn;

	cudnnPoolingDescriptor_t poolingDesc;
	cudnnPoolingMode_t poolingMode;
	cudnnNanPropagation_t maxpoolingNanOpt;
	int windowHeight;
	int windowWidth;
	int verticalPadding;
	int horizontalPadding;
	int verticalStride;
	int horizontalStride;

	void initPoolingDesc();

public:
	PoolingLayer(std::string name_, LayerShape shape_, cudnnHandle_t hCudnn_,
		int windowHeight_ = 2, int windowWidth_ = 2, 
		int verticalStride_ = 2, int horizontalStride_ = 2, 
		int verticalPadding_ = 0, int horizontalPadding_ = 0, 
		cudnnPoolingMode_t poolingMode_ = POOL_MAX_DET, 
		cudnnNanPropagation_t maxpoolingNanOpt_ = NAN_NOT_PROPAGATE);

	void init();

	void forward();
	void backward(float learning_rate, bool last);

	~PoolingLayer();
};
