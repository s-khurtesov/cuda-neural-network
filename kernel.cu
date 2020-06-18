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
        LayerShape shape1(100, 2, 30);
        LayerShape shape2(100, 30, 1);

        NeuralNetwork nn;
        nn.addLayer(new ConvolutionLayer("Convolution_1", shape1, g_hCudnn));
        nn.addLayer(new ConvolutionLayer("Convolution_2", shape2, g_hCudnn));

        Tensor* p_x = nn.getX();
        Tensor* p_dy = nn.getY();
        Tensor y(p_dy->N, p_dy->C, p_dy->H, p_dy->W);

        CHECK_CUDA(cudaDeviceSynchronize());

        y = nn.forward(p_x);

        for (int i = 0; i < y.size(); i++)
            p_dy->data[i] = 0.5f - y.data[i];

        nn.backward(p_dy, 0.01f);

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CleanCudnn(&g_hCudnn);
    CleanCuda();

    return 0;
}
