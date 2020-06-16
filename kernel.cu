#include "common.cuh"

cudnnHandle_t g_hCudnn;

int main()
{
    InitCuda();
    InitCudnn(&g_hCudnn);

    CleanCudnn(&g_hCudnn);
    CleanCuda();

    return 0;
}
