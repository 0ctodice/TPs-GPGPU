/*
 * TP 1 - Premiers pas en CUDA
 * --------------------------
 * Ex 2: Addition de vecteurs
 *
 * File: student.cu
 * Author: Maxime MARIA
 */

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		int idx = threadIdx.x;
		int blx = blockIdx.x;
		int id = blx * blockDim.x + idx;
		if (n > id)
			dev_res[id] = dev_a[id] + dev_b[id];
	}

	void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU

		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		// Allocate arrays on device (input and ouput)

		const size_t bytes = size * sizeof(int);
		std::cout << "Allocating input (3 arrays): "
				  << ((3 * bytes) >> 20) << " MB on Device" << std::endl;

		chrGPU.start();

		// Allocate GPU memory

		cudaMalloc((void **)&dev_a, bytes);
		cudaMalloc((void **)&dev_b, bytes);
		cudaMalloc((void **)&dev_res, bytes);

		// Transfert the array to the GPU

		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

		int block_size = ((int)size / 1024) + 1;

		sumArraysCUDA<<<block_size, 1024>>>(size, dev_a, dev_b, dev_res);

		cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);

		// Free britney

		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);

		chrGPU.stop();

		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl
				  << std::endl;
	}
}
