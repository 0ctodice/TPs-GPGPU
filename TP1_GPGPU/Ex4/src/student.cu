/*
 * TP 1 - Premiers pas en CUDA
 * --------------------------
 * Ex 3: Filtre d'images sepia
 *
 * File: student.cu
 * Author: Maxime MARIA
 */

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void matrixAdditionCUDA(const int *const dev_inputA, const int *const dev_inputB, const uint width, const uint height, int *const dev_output)
	{
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		int id = x + y * width;

		if (id >= width * height)
		{
			return;
		}

		dev_output[id] = dev_inputA[id] + dev_inputB[id];
	}

	void studentJob(const std::vector<int> &inputA, const std::vector<int> &inputB, const uint width, const uint height, std::vector<int> &output)
	{
		ChronoGPU chrGPU;

		// // 2 arrays for GPU
		int *dev_inputA = NULL;
		int *dev_inputB = NULL;
		int *dev_output = NULL;

		const int size = width * height;

		const size_t bytes = size * sizeof(int);

		chrGPU.start();

		cudaMalloc((void **)&dev_inputA, bytes);
		cudaMalloc((void **)&dev_inputB, bytes);
		cudaMalloc((void **)&dev_output, bytes);

		cudaMemcpy(dev_inputA, &inputA[0], bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_inputB, &inputB[0], bytes, cudaMemcpyHostToDevice);

		int threads_size = 32;

		dim3 block_size = dim3((width / threads_size) + 1, (height / threads_size) + 1, 1);
		dim3 nb_threads = dim3(threads_size, threads_size, 1);

		matrixAdditionCUDA<<<block_size, nb_threads>>>(dev_inputA, dev_inputB, width, height, dev_output);

		cudaMemcpy(&output[0], dev_output, bytes, cudaMemcpyDeviceToHost);

		cudaFree(dev_inputA);
		cudaFree(dev_inputB);
		cudaFree(dev_output);

		chrGPU.stop();

		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl
				  << std::endl;
	}
}
