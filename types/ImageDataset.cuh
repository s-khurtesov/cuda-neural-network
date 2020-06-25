#pragma once

#include "../common.cuh"
#include "../stb_image.h"
#include "Tensor.cuh"
#include "LayerShape.cuh"
#include <vector>
#include <iostream>
#include <filesystem>

#define PATH_PARASITIZED "D:\\Projects\\Visual Studio\\cuda-neural-network\\cell_images\\Parasitized_resized_PIL_50x50"
#define PATH_UNINFECTED "D:\\Projects\\Visual Studio\\cuda-neural-network\\cell_images\\Uninfected_resized_PIL_50x50"

enum class DatasetType
{
    TRAIN,
    TEST
};

class ImageDataset {
private:
    DatasetType type;
	std::vector<Tensor> input;
	std::vector<Tensor> target;
	int numberOfBatches;

    void load(bool shuffle)
    {
        std::filesystem::path path_parasitised(PATH_PARASITIZED);
        std::filesystem::path path_uninfected(PATH_UNINFECTED);
        std::vector<std::filesystem::path> files_parasitised, files_uninfected;
        const int batch_size = input[0].N;

        // Calculate number of element of each class in one batch
        const int n_parasitised = batch_size / 2;
        const int n_uninfected = batch_size - n_parasitised;

        assert(n_parasitised > 0);
        assert(n_uninfected > 0);
        assert(batch_size == n_parasitised + n_uninfected);

        // Get paths os all images
        assert(std::filesystem::exists(path_parasitised));
        assert(std::filesystem::exists(path_uninfected));

        for (const auto& dirEntry : std::filesystem::directory_iterator(path_parasitised)) {
            if (dirEntry.exists() && dirEntry.file_size() && dirEntry.is_regular_file()) {
                files_parasitised.push_back(dirEntry.path());
            }
        }

        for (const auto& dirEntry : std::filesystem::directory_iterator(path_uninfected)) {
            if (dirEntry.exists() && dirEntry.file_size() && dirEntry.is_regular_file()) {
                files_uninfected.push_back(dirEntry.path());
            }
        }
        assert(files_parasitised.size() >= n_parasitised * numberOfBatches);
        assert(files_uninfected.size() >= n_uninfected * numberOfBatches);

        // Fit to number of element of each class (if VALIDATION begin at end)
        if (type == DatasetType::TEST) {
            files_parasitised.erase(files_parasitised.begin(), files_parasitised.end() - (int)(n_parasitised * numberOfBatches * 1.1f));
            files_uninfected.erase(files_uninfected.begin(), files_uninfected.end() - (int)(n_uninfected * numberOfBatches * 1.1f));
        }
        else {
            files_parasitised.erase(files_parasitised.begin() + (int)(n_parasitised * numberOfBatches * 1.1f), files_parasitised.end());
            files_uninfected.erase(files_uninfected.begin() + (int)(n_uninfected * numberOfBatches * 1.1f), files_uninfected.end());
        }
        assert(files_parasitised.size() >= n_parasitised * numberOfBatches);
        assert(files_uninfected.size() >= n_uninfected * numberOfBatches);

        // Load images
        std::filesystem::path path;
        const int C = input[0].C;
        const int H = input[0].H;
        const int W = input[0].W;
        int parasitised_loaded = 0;
        int uninfected_loaded = 0;
        if (!shuffle) {
            for (int batch = 0; batch < numberOfBatches; batch++) {
                // First half parasited: n = { 0, ..., n_parasitised - 1}
                for (int n = 0; n < n_parasitised; n++) {
                    path = files_parasitised[batch * n_parasitised + n];

                    assert(read_png(path.string(), input[batch].data + n * C * H * W, C, H, W));
                    //input[batch].show(path.string().c_str(), 2, 3, 50, 50);
                    int nonzero = std::count_if(input[batch].data + n * C * H * W, input[batch].data + (n + 1) * C * H * W, [](float x) {return x > 0.0f; });
                    //printf("[parasited] Nonzeros %d\n", nonzero);
                    assert(nonzero > 0);

                    target[batch].data[n] = 1.0f;
                    parasitised_loaded++;
                }
                // Second half uninfected: n = { 0, ..., n_uninfected - 1}
                for (int n = 0; n < n_uninfected; n++) {
                    path = files_uninfected[batch * n_uninfected + n];

                    assert(read_png(path.string(), input[batch].data + (n + n_parasitised) * C * H * W, C, H, W));
                    //input[batch].show(path.string().c_str(), 2, 3, 50, 50);
                    int nonzero = std::count_if(input[batch].data + (n + n_parasitised) * C * H * W, input[batch].data + ((n + n_parasitised) + 1) * C * H * W, [](float x) {return x > 0.0f; });
                    //printf("[uninfected] Nonzeros %d\n", nonzero);
                    assert(nonzero > 0);

                    target[batch].data[n + n_parasitised] = 0.0f;
                    uninfected_loaded++;
                }
            }
        }
        else {
            for (int batch = 0; batch < numberOfBatches; batch++) {
                for (int n = 0; n < batch_size; n++) {
                    if (((float)rand() / RAND_MAX) > 0.5f && !files_parasitised.empty()) {
                    Parasitized:
                        // Parasitised
                        int i = rand() % files_parasitised.size();
                        path = files_parasitised[i];

                        assert(read_png(path.string(), input[batch].data + n * C * H * W, C, H, W));
                        files_parasitised.erase(files_parasitised.begin() + i);

                        target[batch].data[n] = 1.0f;
                        parasitised_loaded++;
                    }
                    else if (!files_uninfected.empty()) {
                        // Uninfected
                        int i = rand() % files_uninfected.size();
                        path = files_uninfected[i];

                        assert(read_png(path.string(), input[batch].data + n * C * H * W, C, H, W));
                        files_uninfected.erase(files_uninfected.begin() + i);

                        target[batch].data[n] = 0.0f;
                        uninfected_loaded++;
                    }
                    else {
                        goto Parasitized;
                    }
                    //printf("%1.0f ", target[batch].data[n]);
                }
            }
        }
        printf("Loaded %d files from \"%s\"\n", parasitised_loaded, path_parasitised.string().c_str());
        printf("Loaded %d files from \"%s\"\n\n", uninfected_loaded, path_uninfected.string().c_str());
    }

    bool read_png(std::string& filename, float* normalized_data, int x_channels, int x_height, int x_width)
    {
        unsigned char* data;
        int channels, height, width;

        data = stbi_load(filename.c_str(), &width, &height, &channels, 0);

        assert(x_channels == channels);
        assert(x_height == height);
        assert(x_width == width);

        if (!(data != nullptr && width > 0 && height > 0))
        {
            std::cout << "Some error" << std::endl;
            if (data)
                stbi_image_free(data);
            return false;
        }

        for (size_t c = 0; c < channels; c++) {
            for (size_t i = 0; i < width * height; i++) {
                normalized_data[c * width * height + i] = (float)data[i * channels + c] / 255.0f;
            }
        }

        //printf("%s\n[read_png] Nonzeros %d\n", filename.c_str(), std::count_if(normalized_data, normalized_data + x_channels * x_height * x_width, [](float x) {return x > 0.0f; }));

        stbi_image_free(data);

        return true;
    }


public:
	ImageDataset(DatasetType type_, int numberOfBatches_, LayerShape& inputShape, LayerShape& outputShape, bool shuffle=false)
		: type(type_), numberOfBatches(numberOfBatches_)
	{
        // Initialize and allocate
		input.resize(numberOfBatches);
		target.resize(numberOfBatches);
		for (Tensor& tensor : input) {
			tensor.init(inputShape.batch_size, inputShape.in_nrns, inputShape.in_nrn_h, inputShape.in_nrn_w);
		}
		for (Tensor& tensor : target) {
			tensor.init(outputShape.batch_size, outputShape.out_nrns, outputShape.out_nrn_h, outputShape.out_nrn_w);
		}

        // Load
        load(shuffle);

        // Check
        for (int b = 0; b < size(); b++) {
            for (int n = 0; n < inputShape.batch_size; n++) {
                int nonzero = std::count_if(getInput(b).data + n * getInput(b).size() / inputShape.batch_size,
                    getInput(b).data + (n + 1) * getInput(b).size() / inputShape.batch_size, [](float x) {return x > 0.0f; });
                //printf("Batch %d, N_%d: %d nonzeros\n", b + 1, n + 1, nonzero);
                assert(nonzero > 0);
            }
        }
	}

	int size()
	{
		return numberOfBatches;
	}

    Tensor& getInput(int i)
    {
        assert(i < size());
        return input[i];
    }

    Tensor& getTarget(int i)
    {
        assert(i < size());
        return target[i];
    }
};
