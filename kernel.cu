#include "common.cuh"

#include "types/Tensor.cuh"
#include "types/LayerShape.cuh"
#include "layers/ConvolutionLayer.cuh"
#include "NeuralNetwork.cuh"

cudnnHandle_t g_hCudnn;

int main()
{
    InitCuda();
    InitCudnn(&g_hCudnn);

    {
        NeuralNetwork nn;
        Tensor x;
        Tensor y;
        Tensor labels;
        int batchSize = 10;

        nn.addLayer(new ConvolutionLayer("Convolution_1", LayerShape(batchSize, 2, 30), g_hCudnn));
        nn.addLayer(new ConvolutionLayer("Convolution_2", LayerShape(batchSize, 30, 1), g_hCudnn));

        nn.init();

        x = nn.getX();
        y = nn.getY();
        labels = y;

        x.randomise();
        for (int n = 0; n < y.N; n++) {
            labels.data[n/* * labels.C*/] = (x.data[n * x.C] + x.data[n * x.C + 1] > 0) ? 1.0f : 0.0f;
        }

        nn.train(x, labels, 100, 0.01f);
    }

    CleanCudnn(&g_hCudnn);
    CleanCuda();

    return 0;
}
