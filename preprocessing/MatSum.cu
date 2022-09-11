# include <stdio.h>
# include <cuda_runtime.h>
// CUDA Kernel
__global__ void MatSumKernel(int *A, int *B, int *C, int *Count, int numYears, int numCountries)
{
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.z * blockIdx.z + threadIdx.z;
    int tot = (gridDim.x * blockIdx.x * gridDim.y * blockIdx.y) * j + gridDim.x * blockIdx.x * i + y;
    int cnt = gridDim.y * blockIdx.y * j + i;
    if ((y < numYears) && (i < numCountries) && (j < numCountries))
    {
        if ((A[tot] != 0) ^ (B[tot] != 0))
        {
            C[tot] = (A[tot] + B[tot])/2;
        }
        else
        {
            C[tot] = A[tot] + B[tot];
        }
        if (C[tot] != 0)
        {
            Count[cnt]++;
        }
    }
}
