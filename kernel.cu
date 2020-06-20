#include "common.cuh"

#include "types/Tensor.cuh"
#include "types/LayerShape.cuh"
#include "layers/ConvolutionLayer.cuh"
#include "layers/ActivationLayer.cuh"
#include "NeuralNetwork.cuh"

#include <time.h>

cublasHandle_t g_hCublas;
cudnnHandle_t g_hCudnn;

void initInput(Tensor& x);
void calcLabels(Tensor& x, Tensor& targets);
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
        int batchSize = 1000;
        int iterations = 100;
        float learning_rate = 0.001f;

        // Build deep neural network model
        nn.addLayer(new ConvolutionLayer(   "1 Convolution",        LayerShape(batchSize, 2, 30),   g_hCudnn, g_hCublas             ));
        nn.addLayer(new ActivationLayer(    "1 ReLU Activation",    LayerShape(batchSize, 30, 30),  g_hCudnn,           ACT_RELU    ));
        nn.addLayer(new ConvolutionLayer(   "2 Convolution",        LayerShape(batchSize, 30, 1),   g_hCudnn, g_hCublas             ));
        nn.addLayer(new ActivationLayer(    "2 Sigmoid Activation", LayerShape(batchSize, 1, 1),    g_hCudnn,           ACT_SIGMOID ));

        // Initialize neural network
        nn.init();

        // Initialize training tensors
        x = nn.getX();
        targets = nn.getY();
        initInput(x);
        calcLabels(x, targets);

        // Train model
        nn.train(x, targets, iterations, learning_rate);

        // Initialize evaluation tensors
        eval_x = nn.getX();
        eval_targets = nn.getY();
        initInput(eval_x);
        calcLabels(eval_x, eval_targets);

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

void initInput(Tensor& x)
{
    srand((unsigned int)time(NULL));
    x.randomise();
}

void calcLabels(Tensor& x, Tensor& targets)
{
    for (int n = 0; n < targets.N; n++) {
        targets.data[n] = (x.data[n * x.C] + x.data[n * x.C + 1] > 0) ? 1.0f : 0.0f;
    }
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
    printf("Accuracy: %f\n", right_predictions ? (float)right_predictions / y.N : 0.0f);
}
