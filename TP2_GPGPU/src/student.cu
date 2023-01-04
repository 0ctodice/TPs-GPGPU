/*
 * TP 2 - Convolution d'images
 * --------------------------
 * Mémoire constante et textures
 *
 * File: student.cu
 * Author: Maxime MARIA
 */

#include "student.hpp"
#include "chronoGPU.hpp"
#include <algorithm>

namespace IMAC
{

	// ================================================== For image comparison
	std::ostream &operator<<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";
		return os;
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 || std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
	// ==================================================

	__constant__ float KERNEL[2048];

	texture<uchar4, 1, cudaReadModeElementType> image_source;

	texture<uchar4, 2, cudaReadModeElementType> image_source2D;

	__global__ void convText2DCUDA(const int imgWidth, const int imgHeight, const int matSize, uchar4 *const dev_image_output)
	{
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= imgWidth || y >= imgHeight)
		{
			return;
		}

		float r = 0.f;
		float g = 0.f;
		float b = 0.f;

		int offset = matSize / 2;

		int j, i, kY, kX, tmpX, tmpY, id;
		float kernel;
		uchar4 pix;

		for (j = -offset; j <= offset; j++)
		{
			kY = j + offset;
			for (i = -offset; i <= offset; i++)
			{
				kX = i + offset;
				kernel = KERNEL[kX + kY * matSize];

				tmpX = max(0, min(imgWidth - 1, x + i));
				tmpY = max(0, min(imgHeight - 1, y + j));

				pix = tex2D(image_source2D, tmpX, tmpY);

				r += kernel * static_cast<float>(pix.x);
				g += kernel * static_cast<float>(pix.y);
				b += kernel * static_cast<float>(pix.z);
			}
		}

		r = max(0.f, min(255.f, r));
		g = max(0.f, min(255.f, g));
		b = max(0.f, min(255.f, b));

		id = x + y * imgWidth;

		dev_image_output[id].x = r;
		dev_image_output[id].y = g;
		dev_image_output[id].z = b;
		dev_image_output[id].w = 255;
	}

	__global__ void convText1DCUDA(const int imgWidth, const int imgHeight, const int matSize, uchar4 *const dev_image_output)
	{
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= imgWidth || y >= imgHeight)
		{
			return;
		}

		float r = 0.f;
		float g = 0.f;
		float b = 0.f;

		int offset = matSize / 2;

		for (int j = -offset; j <= offset; j++)
		{
			int kY = j + offset;
			for (int i = -offset; i <= offset; i++)
			{
				int kX = i + offset;
				float kernel = KERNEL[kX + kY * matSize];

				int tmpX = max(0, min(imgWidth - 1, x + i));
				int tmpY = max(0, min(imgHeight - 1, y + j));

				int tmpId = tmpX + tmpY * imgWidth;

				uchar4 pix = tex1Dfetch(image_source, tmpId);

				r += kernel * static_cast<float>(pix.x);
				g += kernel * static_cast<float>(pix.y);
				b += kernel * static_cast<float>(pix.z);
			}
		}

		r = max(0.f, min(255.f, r));
		g = max(0.f, min(255.f, g));
		b = max(0.f, min(255.f, b));

		int id = x + y * imgWidth;

		dev_image_output[id].x = r;
		dev_image_output[id].y = g;
		dev_image_output[id].z = b;
		dev_image_output[id].w = 255;
	}

	__global__ void convConstCUDA(const int imgWidth, const int imgHeight, const int matSize, const uchar4 *const dev_image_input, uchar4 *const dev_image_output)
	{
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= imgWidth || y >= imgHeight)
		{
			return;
		}

		float r = 0.f;
		float g = 0.f;
		float b = 0.f;

		int offset = matSize / 2;

		for (int j = -offset; j <= offset; j++)
		{
			int kY = j + offset;
			for (int i = -offset; i <= offset; i++)
			{
				int kX = i + offset;
				float kernel = KERNEL[kX + kY * matSize];

				int tmpX = max(0, min(imgWidth - 1, x + i));
				int tmpY = max(0, min(imgHeight - 1, y + j));

				int tmpId = tmpX + tmpY * imgWidth;

				r += kernel * static_cast<float>(dev_image_input[tmpId].x);
				g += kernel * static_cast<float>(dev_image_input[tmpId].y);
				b += kernel * static_cast<float>(dev_image_input[tmpId].z);
			}
		}

		r = max(0.f, min(255.f, r));
		g = max(0.f, min(255.f, g));
		b = max(0.f, min(255.f, b));

		int id = x + y * imgWidth;

		dev_image_output[id].x = r;
		dev_image_output[id].y = g;
		dev_image_output[id].z = b;
		dev_image_output[id].w = 255;
	}

	__global__ void convNaifCUDA(const int imgWidth, const int imgHeight, const int matSize, const uchar4 *const dev_image_input, uchar4 *const dev_image_output, const float *const matConv)
	{
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x >= imgWidth || y >= imgHeight)
		{
			return;
		}

		float r = 0.f;
		float g = 0.f;
		float b = 0.f;

		int offset = matSize / 2;

		for (int j = -offset; j <= offset; j++)
		{
			int kY = j + offset;
			for (int i = -offset; i <= offset; i++)
			{
				int kX = i + offset;
				float kernel = matConv[kX + kY * matSize];

				int tmpX = max(0, min(imgWidth - 1, x + i));
				int tmpY = max(0, min(imgHeight - 1, y + j));

				int tmpId = tmpX + tmpY * imgWidth;

				r += kernel * static_cast<float>(dev_image_input[tmpId].x);
				g += kernel * static_cast<float>(dev_image_input[tmpId].y);
				b += kernel * static_cast<float>(dev_image_input[tmpId].z);
			}
		}

		r = max(0.f, min(255.f, r));
		g = max(0.f, min(255.f, g));
		b = max(0.f, min(255.f, b));

		int id = x + y * imgWidth;

		dev_image_output[id].x = r;
		dev_image_output[id].y = g;
		dev_image_output[id].z = b;
		dev_image_output[id].w = 255;
	}

	void studentJob(const std::vector<uchar4> &inputImg,	   // Input image
					const uint imgWidth, const uint imgHeight, // Image size
					const std::vector<float> &matConv,		   // Convolution matrix (square)
					const uint matSize,						   // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU,	   // Just for comparison
					std::vector<uchar4> &output				   // Output image
	)
	{
		ChronoGPU chrGPU;
		chrGPU.start();

		uchar4 *dev_image_input = NULL;
		uchar4 *dev_image_output = NULL;

		const int size = imgWidth * imgHeight;

		const size_t bytes = size * sizeof(uchar4);

		int threads_size = 32;

		dim3 block_size = dim3((imgWidth / threads_size) + 1, (imgHeight / threads_size) + 1, 1);
		dim3 nb_threads = dim3(threads_size, threads_size, 1);

		cudaMalloc((void **)&dev_image_output, bytes);

		// ================================================== PARTIE NAIF
		// cudaMalloc((void **)&dev_image_input, bytes);
		// cudaMemcpy(dev_image_input, &inputImg[0], bytes, cudaMemcpyHostToDevice);
		// float *dev_mat_input = NULL;
		// cudaMalloc((void **)&dev_mat_input, matSize * matSize * sizeof(float));
		// cudaMemcpy(dev_mat_input, &matConv[0], matSize * matSize * sizeof(float), cudaMemcpyHostToDevice);
		// convNaifCUDA<<<block_size, nb_threads>>>(imgWidth, imgHeight, matSize, dev_image_input, dev_image_output, dev_mat_input);
		// cudaFree(dev_mat_input);

		// ================================================== PARTIE NOYAU CONSTANT

		// cudaMalloc((void **)&dev_image_input, bytes);
		// cudaMemcpy(dev_image_input, &inputImg[0], bytes, cudaMemcpyHostToDevice);
		// cudaMemcpyToSymbol(KERNEL, &matConv[0], matSize * matSize * sizeof(float), 0, cudaMemcpyHostToDevice);
		// convConstCUDA<<<block_size, nb_threads>>>(imgWidth, imgHeight, matSize, dev_image_input, dev_image_output);

		// ================================================== PARTIE TEXTURE 1D

		// cudaMalloc((void **)&dev_image_input, bytes);
		// cudaMemcpy(dev_image_input, &inputImg[0], bytes, cudaMemcpyHostToDevice);
		// cudaBindTexture(0, image_source, dev_image_input, bytes);
		// cudaMemcpyToSymbol(KERNEL, &matConv[0], matSize * matSize * sizeof(float), 0, cudaMemcpyHostToDevice);
		// convText1DCUDA<<<block_size, nb_threads>>>(imgWidth, imgHeight, matSize, dev_image_output);
		// cudaUnbindTexture(image_source);

		// ================================================== PARTIE TEXTURE 2D
		size_t dev_pitch;

		cudaMallocPitch(&dev_image_input, &dev_pitch, imgWidth * sizeof(uchar4), imgHeight);
		cudaMemcpy2D(dev_image_input, dev_pitch, &inputImg[0], imgWidth * sizeof(uchar4), imgWidth * sizeof(uchar4), imgHeight, cudaMemcpyHostToDevice);
		cudaBindTexture2D(0, image_source2D, dev_image_input, imgWidth, imgHeight, dev_pitch);
		cudaMemcpyToSymbol(KERNEL, &matConv[0], matSize * matSize * sizeof(float), 0, cudaMemcpyHostToDevice);
		convText2DCUDA<<<block_size, nb_threads>>>(imgWidth, imgHeight, matSize, dev_image_output);
		cudaUnbindTexture(image_source2D);

		// ==================================================

		cudaMemcpy(&output[0], dev_image_output, bytes, cudaMemcpyDeviceToHost);

		cudaFree(dev_image_input);
		cudaFree(dev_image_output);

		chrGPU.stop();

		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl
				  << std::endl;

		compareImages(resultCPU, output);
	}
}
