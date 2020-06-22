#pragma once

#include "../common.cuh"
#include <assert.h>
#include <random>
#include <functional>

#define SHOW_MAX_N 2
#define SHOW_MAX_C 2
#define SHOW_MAX_H 2
#define SHOW_MAX_W 2

class Descriptor {
protected:
	bool initialized = false;

public:
	cudnnTensorFormat_t format;
	const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
	float* data;

	int N;
	int C;
	int H;
	int W;

	Descriptor()
	{
		format = CUDNN_TENSOR_NCHW;
		data = NULL;
		N = C = H = W = 0;
	}

	virtual void init(int N_, int C_, int H_, int W_, cudnnTensorFormat_t format_ = CUDNN_TENSOR_NCHW) = 0;

	int size() { return N * C * H * W; }

	void normalDistribution(float threshold = 1.0f)
	{
		assert(data);

		std::default_random_engine generator;
		std::normal_distribution<float> normal_distribution(0.0f, 1.0f);

		for (int i = 0; i < size(); i++) {
			data[i] = normal_distribution(generator) * threshold;
		}
	}

	void randomise(float scale = 1.0f)
	{
		assert(scale > 0.0f);
		assert(data);

		for (int i = 0; i < size(); i++) {
			data[i] = rand() / ((float)RAND_MAX / (2.0f * scale)) - scale;
		}
	}

	void fill(float val)
	{
		assert(data);

		for (int i = 0; i < size(); i++) {
			data[i] = val;
		}
	}

	void show(const char* descr, 
		int max_n = SHOW_MAX_N, int max_c = SHOW_MAX_C, 
		int max_h = SHOW_MAX_H, int max_w = SHOW_MAX_W)
	{
		assert(data);

		int tail_n = max_n / 2;
		int head_n = max_n - tail_n;
		int tail_c = max_c / 2;
		int head_c = max_c - tail_c;
		int tail_h = max_h / 2;
		int head_h = max_h - tail_h;
		int tail_w = max_w / 2;
		int head_w = max_w - tail_w;

		printf("%s:\n", descr);
		for (int n = 0; n < N; n++) {
			if (n < head_n || n >= N - tail_n) {
				printf("    N_%d:\n", n);
				for (int c = 0; c < C; c++) {
					if (c < head_c || c >= C - tail_c) {
						printf("        C_%d:\n", c);
						for (int h = 0; h < H; h++) {
							if (h < head_h || h >= H - tail_h) {
								for (int w = 0; w < W; w++) {
									if (w < head_w || w >= W - tail_w) {
										printf("%11.8f ", data[n * C + c * H + h * W + w]);
									}
									else {
										printf("... ");
										w = W - tail_w - 1;
									}
								}
								printf("\n");
							}
							else {
								printf("            ...  ...\n");
								h = H - tail_h - 1;
							}
						}
					}
					else {
						printf("        ...  ...  ...\n");
						c = C - tail_c - 1;
					}
				}
			}
			else {
				printf("    ...  ...  ...  ...\n");
				n = N - tail_n - 1;
			}
		}
		putchar('\n');
	}

	template<typename F>
	void forEach(F function)
	{
		for (int i = 0; i < size(); i++) {
			data[i] = function(data[i]);
		}
	}
};
