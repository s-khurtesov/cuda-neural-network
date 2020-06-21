#include "common.cuh"

#include "types/Tensor.cuh"
#include "types/LayerShape.cuh"
#include "layers/ConvolutionLayer.cuh"
#include "layers/ActivationLayer.cuh"
#include "layers/PoolingLayer.cuh"
#include "NeuralNetwork.cuh"

#include <time.h>
#include <iostream>
#include <filesystem>

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
}

#define PATH_PARASITIZED "D:\\Projects\\Visual Studio\\cuda-neural-network\\cell_images\\Parasitized_resized_PIL_50x50"
#define PATH_UNINFECTED "D:\\Projects\\Visual Studio\\cuda-neural-network\\cell_images\\Uninfected_resized_PIL_50x50"

cublasHandle_t g_hCublas;
cudnnHandle_t g_hCudnn;

void initInputAndLabels(Tensor& x, Tensor& targets, bool evaluation = false);
bool read_png(std::string filename, float* normalized_data, int x_channels, int x_height, int x_width);
void stats(Tensor& x, Tensor& y, Tensor& targets);

int main()
{
    // Initialize CUDA device, cuBLAS and cuDNN handles
    InitCuda();
    InitCublas(&g_hCublas);
    InitCudnn(&g_hCudnn);

    // Objects (like Tensor) must be destructed before destroying CUDA handles
    {
        NeuralNetwork nn;
        Tensor x;
        Tensor targets;
        Tensor eval_x;
        Tensor eval_targets;
        Tensor y;
        int batchSize = 20;
        int channels = 3;
        int height = 50;
        int width = 50;
        int iterations = 1000;
        float learning_rate = 0.01f;

        // Set shapes
        LayerShape l1(batchSize, channels,      channels * 3,   height,         width,          height / 3,     width / 3);
        LayerShape l2(batchSize, l1.out_nrns,   l1.out_nrns,    l1.out_nrn_h,   l1.out_nrn_w,   l1.out_nrn_h,   l1.out_nrn_w);
        LayerShape l3(batchSize, l2.out_nrns,   1,              l2.out_nrn_h,   l2.out_nrn_w,   1,              1);
        LayerShape l4(batchSize, l3.out_nrns,   l3.out_nrns,    l3.out_nrn_h,   l3.out_nrn_w,   l3.out_nrn_h,   l3.out_nrn_w);

        // Build deep neural network model
        nn.addLayer(new ConvolutionLayer(   "1 Convolution",        l1,  g_hCudnn, g_hCublas             ));
        nn.addLayer(new ActivationLayer(    "1 ReLU Activation",    l2,  g_hCudnn,           ACT_RELU    ));
        nn.addLayer(new ConvolutionLayer(   "2 Convolution",        l3,  g_hCudnn, g_hCublas             ));
        nn.addLayer(new ActivationLayer(    "2 Sigmoid Activation", l4,  g_hCudnn,           ACT_SIGMOID ));

        // Initialize neural network
        nn.init();

        // Initialize training tensors
        x = nn.getX();
        targets = nn.getY();
        initInputAndLabels(x, targets);

        printf("x non zero: %d, targets non zero: %d\n", x.nonZeroCount(), targets.nonZeroCount());

        // Train model
        nn.train(x, targets, iterations, learning_rate, 0.5f);

        // Initialize evaluation tensors
        eval_x = nn.getX();
        eval_targets = nn.getY();
        initInputAndLabels(eval_x, eval_targets, true);

        printf("eval_x non zero: %d, eval_targets non zero: %d\n", eval_x.nonZeroCount(), eval_targets.nonZeroCount());

        // Evaluate model
        y = nn.forward(eval_x);

        // Print statistics
        stats(eval_x, y, eval_targets);
    }

    // Destroy cuBLAS and cuDNN handles, reset CUDA device
    CleanCudnn(&g_hCudnn);
    CleanCublas(&g_hCublas);
    CleanCuda();

    return 0;
}

void initInputAndLabels(Tensor& x, Tensor& targets, bool evaluation)
{
    std::filesystem::path path_parasitised(PATH_PARASITIZED);
    std::filesystem::path path_uninfected(PATH_UNINFECTED);
    std::vector<std::filesystem::path> files_parasitised, files_uninfected;
    const int n_parasitised = x.N / 2;
    const int n_uninfected = x.N - n_parasitised;
    int n = 0;

    assert(n_parasitised > 0);
    assert(n_uninfected > 0);
    assert(x.N == n_parasitised + n_uninfected);

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

    if (evaluation) {
        files_parasitised.erase(files_parasitised.begin(), files_parasitised.end() - n_parasitised);
        files_uninfected.erase(files_uninfected.begin(), files_uninfected.end() - n_uninfected);
    }
    else {
        files_parasitised.erase(files_parasitised.begin() + n_parasitised, files_parasitised.end());
        files_uninfected.erase(files_uninfected.begin() + n_uninfected, files_uninfected.end());
    }
    assert(files_parasitised.size() == n_parasitised);
    assert(files_uninfected.size() == n_uninfected);

    for (const std::filesystem::path& path : files_parasitised) {
        //printf("Loading \"%s\"...\n", path.filename().string().c_str());
        assert(read_png(path.string(), x.data + n * x.C * x.H * x.W, x.C, x.H, x.W));
        targets.data[n] = 1.0f;
        n++;
    }
    printf("Loaded %d files from \"%s\"\n", n, path_parasitised.string().c_str());
    assert(n == n_parasitised);

    n = 0;
    for (const std::filesystem::path& path : files_uninfected) {
        //printf("Loading \"%s\"...\n", path.filename().string().c_str());
        assert(read_png(path.string(), x.data + (n_parasitised + n) * x.C * x.H * x.W, x.C, x.H, x.W));
        targets.data[n_parasitised + n] = 0.0f;
        n++;
    }
    printf("Loaded %d files from \"%s\"\n\n", n, path_uninfected.string().c_str());
    assert(n == n_uninfected);

    //x.show("x", 1, 2, 30, 30);
}

void stats(Tensor& x, Tensor& y, Tensor& targets)
{
    int right_predictions = 0;
    for (int n = 0; n < targets.N; n++) {
        float prediction = (y.data[n] > 0.5f) ? 1.0f : 0.0f;
        if (prediction == targets.data[n]) {
            right_predictions++;
        }
    }
    printf("Accuracy: %f\n", (float)right_predictions / y.N);
}

bool read_png(std::string filename, float* normalized_data, int x_channels, int x_height, int x_width)
{
    unsigned char* data;
    int channels, height, width;

    data = stbi_load(filename.c_str(), &width, &height, &channels, 0);

    assert(x_channels == channels);
    assert(x_height == height);
    assert(x_width == width);

    if (data != nullptr && width > 0 && height > 0)
    {
        //std::cout << "width: " << width << ", heigth: " << height << ", channels: " << channels << std::endl;
        //if (channels == 3)
        //{
        //    std::cout << "First pixel: RGB "
        //        << static_cast<int>(data[0]) << " "
        //        << static_cast<int>(data[1]) << " "
        //        << static_cast<int>(data[2]) << std::endl;
        //}
        //else if (channels == 4)
        //{
        //    std::cout << "First pixel: RGBA "
        //        << static_cast<int>(data[0]) << " "
        //        << static_cast<int>(data[1]) << " "
        //        << static_cast<int>(data[2]) << " "
        //        << static_cast<int>(data[3]) << std::endl;
        //}
    }
    else
    {
        std::cout << "Some error" << std::endl;
        if (data)
            stbi_image_free(data);
        return false;
    }

    for (size_t c = 0; c < channels; c++) {
        for (size_t i = 0; i < width * height; i++) {
            normalized_data[c * width * height + i] = (float)data[i * channels + c] / 255.0f;
            //printf("normalized_data[%d] = data[%d] / 255 = %f\n", c * width * height + i, i * channels + c, normalized_data[c * width * height + i]);
        }
    }

    stbi_image_free(data);

    return true;
}
