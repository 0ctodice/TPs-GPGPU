/*
 * TP 3 - Réduction CUDA
 * --------------------------
 * Mémoire paratagée, synchronisation, optimisation
 *
 * File: student.cu
 * Author: Maxime MARIA
 */

#include "student.hpp"

namespace IMAC
{
	// ==================================================== EX 1
	__global__ void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];

		int idx = threadIdx.x;
		int blx = blockIdx.x;
		int myId = blx * blockDim.x + idx;

		sharedMemory[idx] = myId < size ? dev_array[myId] : 0;

		__syncthreads();

		for (int i = 1; i < blockDim.x; i *= 2)
		{
			int id = idx * 2 * i;
			if (id < blockDim.x)
			{
				uint maxShared = umax(sharedMemory[id], sharedMemory[id + i]);
				sharedMemory[id] = maxShared;
			}
			__syncthreads();
		}
		if (idx == 0)
		{
			dev_partialMax[blx] = sharedMemory[0];
		}
	}
	// ==================================================== EX 2
	__global__ void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];

		int idx = threadIdx.x;
		int blx = blockIdx.x;
		int myId = blx * blockDim.x + idx;

		sharedMemory[idx] = myId < size ? dev_array[myId] : 0;

		__syncthreads();

		for (int i = (blockDim.x / 2); i > 0; i /= 2)
		{
			if (idx < i)
			{
				uint maxShared = umax(sharedMemory[idx], sharedMemory[idx + i]);
				sharedMemory[idx] = maxShared;
			}
			__syncthreads();
		}

		if (idx == 0)
		{
			dev_partialMax[blx] = sharedMemory[0];
		}
	}
	// ==================================================== EX 3
	__global__ void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];

		int idx = threadIdx.x;
		int blx = blockIdx.x;
		int myId = blx * blockDim.x * 2 + idx;

		if (myId < size)
		{
			if (myId + blockDim.x / 2 < size)
			{
				sharedMemory[idx] = umax(dev_array[myId], dev_array[myId + blockDim.x]);
			}
			else
			{
				sharedMemory[idx] = dev_array[myId];
			}
		}
		else
		{
			sharedMemory[idx] = 0;
		}

		__syncthreads();

		for (int i = (blockDim.x / 2); i > 0; i /= 2)
		{
			if (idx < i)
			{
				uint maxShared = umax(sharedMemory[idx], sharedMemory[idx + i]);
				sharedMemory[idx] = maxShared;
			}
			__syncthreads();
		}

		if (idx == 0)
		{
			dev_partialMax[blx] = sharedMemory[0];
		}
	}
	// ==================================================== EX 4
	__global__ void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];

		int idx = threadIdx.x;
		int blx = blockIdx.x;
		int myId = blx * blockDim.x * 2 + idx;
		int i = 0;

		if (myId < size)
		{
			if (myId + blockDim.x / 2 < size)
			{
				sharedMemory[idx] = umax(dev_array[myId], dev_array[myId + blockDim.x]);
			}
			else
			{
				sharedMemory[idx] = dev_array[myId];
			}
		}
		else
		{
			sharedMemory[idx] = 0;
		}

		__syncthreads();

		for (i = (blockDim.x / 2); i > 0; i /= 2)
		{
			if (i < 32)
			{
				break;
			}

			if (idx < i)
			{
				uint maxShared = umax(sharedMemory[idx], sharedMemory[idx + i]);
				sharedMemory[idx] = maxShared;
			}
			__syncthreads();
		}

		volatile uint *vsm = sharedMemory;

		if (idx < i)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + i]);
			i /= 2;
		}
		if (idx < i)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + i]);
			i /= 2;
		}
		if (idx < i)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + i]);
			i /= 2;
		}
		if (idx < i)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + i]);
			i /= 2;
		}
		if (idx < i)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + i]);
			i /= 2;
		}
		if (idx < i)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + i]);
		}

		if (idx == 0)
		{
			dev_partialMax[blx] = sharedMemory[0];
		}
	}
	// ==================================================== EX 5
	template <uint T>
	__global__ void maxReduce_ex5(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];

		int idx = threadIdx.x;
		int blx = blockIdx.x;
		int myId = blx * blockDim.x * 2 + idx;

		if (myId < size)
		{
			if (myId + blockDim.x / 2 < size)
			{
				sharedMemory[idx] = umax(dev_array[myId], dev_array[myId + blockDim.x]);
			}
			else
			{
				sharedMemory[idx] = dev_array[myId];
			}
		}
		else
		{
			sharedMemory[idx] = 0;
		}

		__syncthreads();

		volatile uint *vsm = sharedMemory;

		if (T >= 1024u && idx < 512u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 512u]);
			__syncthreads();
		}

		if (T >= 512u && idx < 256u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 256u]);
			__syncthreads();
		}

		if (T >= 256u && idx < 128u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 128u]);
			__syncthreads();
		}

		if (T >= 128u && idx < 64u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 64u]);
			__syncthreads();
		}

		if (idx < 32u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 32u]);
		}

		if (idx < 16u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 16u]);
		}

		if (idx < 8u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 8u]);
		}

		if (idx < 4u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 4u]);
		}

		if (idx < 2u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 2u]);
		}

		if (idx < 1u)
		{
			vsm[idx] = umax(vsm[idx], vsm[idx + 1u]);
		}

		if (idx == 0)
		{
			dev_partialMax[blx] = sharedMemory[0];
		}
	}

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
	{
		uint *dev_array = NULL;
		const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR(cudaMalloc((void **)&dev_array, bytes));
		// Copy data from host to device
		HANDLE_ERROR(cudaMemcpy(dev_array, array.data(), bytes, cudaMemcpyHostToDevice));

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);

		std::cout << " -> Done: ";
		printTiming(timing1);
		compare(res1, resCPU); // Compare results

		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);

		std::cout << " -> Done: ";
		printTiming(timing2);
		compare(res2, resCPU);

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);

		std::cout << " -> Done: ";
		printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);

		std::cout << " -> Done: ";
		printTiming(timing4);
		compare(res4, resCPU);

		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);

		std::cout << " -> Done: ";
		printTiming(timing5);
		compare(res5, resCPU);

		// Free array on GPU
		cudaFree(dev_array);
	}

	void printTiming(const float2 timing)
	{
		std::cout << (timing.x < 1.f ? /*1e3f **/ timing.x : timing.x) << " us on device and ";
		std::cout << (timing.y < 1.f ? /*1e3f **/ timing.y : timing.y) << " us on host." << std::endl;
	}

	void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
