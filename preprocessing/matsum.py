import numpy as np
from pycuda import gpuarray, autoinit
import pycuda.driver as cuda
from pycuda.tools import DeviceData
from pycuda.tools import OccupancyRecord as occupancy
from pycuda.compiler import SourceModule
from matplotlib import pyplot as plt

def matsum_gpu(A, B):
    """
    GPU evaluation of (A+B)/2 for adjacency matrix and count of
    n. years the exchange between countries i and j differs from 0.
    """
    num_c, num_y = A.shape[1], A.shape[0]
    C = np.zeros((num_y, num_c, num_c), dtype=np.int32)
    Count = np.zeros((num_c, num_c), dtype=np.int32)

    a_gpu = gpuarray.to_gpu(A)
    b_gpu = gpuarray.to_gpu(B)
    c_gpu = gpuarray.to_gpu(C)
    count_gpu = gpuarray.to_gpu(Count)

    '''
    cudaKernel = """
        __global__ void MatSumKernel(const int *A, const int *B, int *C, int *Count, int numYears, int numCountries)
    {
        int y = blockDim.x * blockIdx.x + threadIdx.x;
        int i = blockDim.y * blockIdx.y + threadIdx.y;
        int j = blockDim.z * blockIdx.z + threadIdx.z;
        if ((y < numYears) && (i < numCountries) && (j < numCountries))
        {
            if ((A[y][i][j] != 0) ^ (B[y][j][i] != 0))
            {
                C[y][i][j] = (A[y][i][j] + B[y][j][i])/2;
            }
            else
            {
                C[y][i][j] = A[y][i][j] + B[y][j][i];
            }
            if (C[y][i][j] != 0)
            {
                Count[i][j]++;
            }
        }
    }
    """
    '''
    cudaCode = open("MatSum.cu","r")
    myCUDACode = cudaCode.read()
    myCode = SourceModule(myCUDACode)
    MatSum = myCode.get_function("MatSumKernel")
    cuBlock = (4, 16, 16)
    cuGrid = (6, 16, 16)
    C = c_gpu.get()
    Count = count_gpu.get()
    return C, Count
