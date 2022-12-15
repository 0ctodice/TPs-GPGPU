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
	__global__ void sepiaCUDA(const int width, const int height, const uchar *const dev_input, uchar *const dev_output)
	{

		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= width || y >= height)
		{
			return;
		}

		int id = x + y * width;

		dev_output[3 * id + 0] = static_cast<uchar>(fminf(255.f, (dev_input[3 * id] * 0.393f + dev_input[3 * id + 1] * 0.769f + dev_input[3 * id + 2] * 0.189f)));
		dev_output[3 * id + 1] = static_cast<uchar>(fminf(255.f, (dev_input[3 * id] * 0.349f + dev_input[3 * id + 1] * 0.686f + dev_input[3 * id + 2] * 0.168f)));
		dev_output[3 * id + 2] = static_cast<uchar>(fminf(255.f, (dev_input[3 * id] * 0.272f + dev_input[3 * id + 1] * 0.534f + dev_input[3 * id + 2] * 0.131f)));
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;

		const int size = width * height * 3;

		const size_t bytes = size * sizeof(uchar);

		chrGPU.start();

		cudaMalloc((void **)&dev_input, bytes);
		cudaMalloc((void **)&dev_output, bytes);

		cudaMemcpy(dev_input, &input[0], bytes, cudaMemcpyHostToDevice);

		int threads_size = 32;

		dim3 block_size = dim3((width / threads_size) + 1, (height / threads_size) + 1, 1);
		dim3 nb_threads = dim3(threads_size, threads_size, 1);

		sepiaCUDA<<<block_size, nb_threads>>>(width, height, dev_input, dev_output);

		cudaMemcpy(&output[0], dev_output, bytes, cudaMemcpyDeviceToHost);

		cudaFree(dev_input);
		cudaFree(dev_output);

		chrGPU.stop();

		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl
				  << std::endl;
	}
}
