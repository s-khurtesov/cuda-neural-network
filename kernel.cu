#include "common.cuh"

#include "types/Tensor.cuh"
#include "types/LayerShape.cuh"
#include "layers/ConvolutionLayer.cuh"
#include "NeuralNetwork.cuh"

cublasHandle_t g_hCublas;
cudnnHandle_t g_hCudnn;

int main()
{
    InitCuda();
    InitCublas(&g_hCublas);
    InitCudnn(&g_hCudnn);

    {
        NeuralNetwork nn;
        Tensor x;
        Tensor labels;
        int batchSize = 10;

        nn.addLayer(new ConvolutionLayer("Convolution_1", LayerShape(batchSize, 2, 30), g_hCudnn, g_hCublas));
        nn.addLayer(new ConvolutionLayer("Convolution_2", LayerShape(batchSize, 30, 1), g_hCudnn, g_hCublas));

        nn.init();

        x = nn.getX();
        labels = nn.getY();

        x.randomise();
        for (int n = 0; n < labels.N; n++) {
            labels.data[n/* * labels.C*/] = (x.data[n * x.C] + x.data[n * x.C + 1] > 0) ? 1.0f : 0.0f;
        }

        x.show("x.randomise()", 4);
        labels.show("labels = x[0] + x[1] > 0", 4);

        nn.train(x, labels, 100, 0.01f);
    }

    CleanCudnn(&g_hCudnn);
    CleanCublas(&g_hCublas);
    CleanCuda();

    return 0;
}
