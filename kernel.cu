#include "common.cuh"

#include "types/Tensor.cuh"
#include "types/LayerShape.cuh"
#include "types/ImageDataset.cuh"
#include "layers/ConvolutionLayer.cuh"
#include "layers/ActivationLayer.cuh"
#include "layers/PoolingLayer.cuh"
#include "layers/DropoutLayer.cuh"
#include "layers/DenseLayer.cuh"
#include "NeuralNetwork.cuh"

#include <time.h>
#include <iostream>

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
}

cublasHandle_t g_hCublas;
cudnnHandle_t g_hCudnn;

void stats(std::vector<Tensor>& results, ImageDataset& dataset);

int main()
{
    // Initialize CUDA device, cuBLAS and cuDNN handles
    InitCuda();
    InitCublas(&g_hCublas);
    InitCudnn(&g_hCudnn);

    srand(1321351351);
    //srand((unsigned int)time(NULL));

    // Objects (like Tensor) must be destructed before destroying CUDA handles
    {
        NeuralNetwork nn;
        int batchSize = 16;
        int evalNumberOfBatches = 3000 / batchSize;
        int numberOfBatches = 10000 / batchSize;
        int channels = 3;
        int height = 50;
        int width = 50;

        int epochs = 10;
        float learning_rate = 0.005f;

        // TODO: Simplify shapes initialization

        // Set shapes
        LayerShape l1(batchSize, channels, 16, 
            height, width,
            height - 2, width - 2);
        LayerShape l2(batchSize, l1.out_nrns, l1.out_nrns,
            l1.out_nrn_h, l1.out_nrn_w,
            l1.out_nrn_h / 2, l1.out_nrn_w / 2);
        LayerShape l3(batchSize, l2.out_nrns, l2.out_nrns * 4,
            l2.out_nrn_h, l2.out_nrn_w,
            l2.out_nrn_h - 2, l2.out_nrn_w - 2);
        LayerShape l4(batchSize, l3.out_nrns, l3.out_nrns,
            l3.out_nrn_h, l3.out_nrn_w,
            l3.out_nrn_h / 2, l3.out_nrn_w / 2);
        LayerShape l5(batchSize, l4.out_nrns, l4.out_nrns * 4,
            l4.out_nrn_h, l4.out_nrn_w,
            l4.out_nrn_h - 2, l4.out_nrn_w - 2);
        LayerShape l6(batchSize, l5.out_nrns, l5.out_nrns,
            l5.out_nrn_h, l5.out_nrn_w,
            l5.out_nrn_h / 2, l5.out_nrn_w / 2);
        LayerShape l7(batchSize, l6.out_nrns, l6.out_nrns / 2,
            l6.out_nrn_h, l6.out_nrn_w,
            l6.out_nrn_h / 2, l6.out_nrn_w / 2);
        LayerShape l8(batchSize, l7.out_nrns, 1,
            l7.out_nrn_h, l7.out_nrn_w,
            1, 1);

        // Activation and dropout
        LayerShape l1_act(batchSize, l1.out_nrns, l1.out_nrns, l1.out_nrn_h, l1.out_nrn_w, l1.out_nrn_h, l1.out_nrn_w);
        LayerShape l3_act(batchSize, l3.out_nrns, l3.out_nrns, l3.out_nrn_h, l3.out_nrn_w, l3.out_nrn_h, l3.out_nrn_w);
        LayerShape l5_act(batchSize, l5.out_nrns, l5.out_nrns, l5.out_nrn_h, l5.out_nrn_w, l5.out_nrn_h, l5.out_nrn_w);
        LayerShape l7_act(batchSize, l7.out_nrns, l7.out_nrns, l7.out_nrn_h, l7.out_nrn_w, l7.out_nrn_h, l7.out_nrn_w);
        LayerShape l8_act(batchSize, l8.out_nrns, l8.out_nrns, l8.out_nrn_h, l8.out_nrn_w, l8.out_nrn_h, l8.out_nrn_w);

        // Build deep neural network model
        nn.addLayer(new ConvolutionLayer("1 Convolution 3x3",   l1,     g_hCudnn, g_hCublas));
        nn.addLayer(new ActivationLayer ("1 ReLU",              l1_act, g_hCudnn, ACT_RELU));
        nn.addLayer(new PoolingLayer    ("2 Max Pooling 2x2",   l2,     g_hCudnn));
        nn.addLayer(new ConvolutionLayer("3 Convolution 3x3",   l3,     g_hCudnn, g_hCublas));
        nn.addLayer(new ActivationLayer ("3 ReLU",              l3_act, g_hCudnn, ACT_RELU));
        nn.addLayer(new PoolingLayer    ("4 Max Pooling 2x2",   l4,     g_hCudnn));
        nn.addLayer(new ConvolutionLayer("5 Convolution 3x3",   l5,     g_hCudnn, g_hCublas));
        nn.addLayer(new ActivationLayer ("5 ReLU",              l5_act, g_hCudnn, ACT_RELU));
        nn.addLayer(new PoolingLayer    ("6 Max Pooling 2x2",   l6,     g_hCudnn));
        nn.addLayer(new DenseLayer      ("7 Dense",             l7,     g_hCublas));
        nn.addLayer(new ActivationLayer ("7 ReLU",              l7_act, g_hCudnn, ACT_RELU));
        nn.addLayer(new DenseLayer      ("8 Dense",             l8,     g_hCublas));
        nn.addLayer(new ActivationLayer ("8 Sigmoid",           l8_act, g_hCudnn, ACT_SIGMOID));

        // Initialize neural network
        nn.init();
        nn.summary();

        {
            // Initialize training dataset
            ImageDataset trainDataset(DatasetType::TRAIN, numberOfBatches, l1, l8_act, true);

            // Train model
            nn.train(trainDataset, epochs, learning_rate, 0.4f, 0.1f);
        }

        // TODO: Allow different batch sizes

        // Initialize evaluation dataset
        srand(time(NULL));
        ImageDataset evalDataset(DatasetType::TEST, evalNumberOfBatches, l1, l8_act, true);

        // Evaluate model
        std::vector<Tensor> results;
        results.resize(evalDataset.size());
        for (int batch = 0; batch < evalDataset.size(); batch++) {
            results[batch] = nn.forward(evalDataset.getInput(batch));
        }

        // Print statistics
        stats(results, evalDataset);
    }

    // Destroy cuBLAS and cuDNN handles, reset CUDA device
    CleanCudnn(&g_hCudnn);
    CleanCublas(&g_hCublas);
    CleanCuda();
    
    return 0;
}

void stats(std::vector<Tensor>& results, ImageDataset& dataset)
{
    const int all_inputs = (int)results.size() * results[0].N;
    int right_ones = 0;
    int right_zeros = 0;
    int all_ones = 0;
    for (int batch = 0; batch < dataset.size(); batch++) {
        Tensor& y = results[batch];
        Tensor& targets = dataset.getTarget(batch);
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
            all_ones += (int)prediction;
        }
    }
    printf("Accuracy: %f\n\tOnes: %f (%d/%d)\n\tZeros: %f (%d/%d)\n", (float)(right_ones + right_zeros) / all_inputs,
        all_ones ? (float)(right_ones) / all_ones : 0.0f, right_ones, all_ones, (float)(right_zeros) / (all_inputs - all_ones), right_zeros, all_inputs - all_ones);
}
