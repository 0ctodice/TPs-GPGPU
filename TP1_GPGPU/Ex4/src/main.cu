/*
 * TP 1 - Premiers pas en CUDA
 * --------------------------
 * Ex 4: Filtre d'images sepia
 *
 * File: main.cpp
 * Author: Thomas Dumont
 */

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cstring>
#include <exception>
#include <algorithm>

#include "student.hpp"
#include "chronoCPU.hpp"

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg)
	{
		std::cerr << "Usage: " << prg << std::endl
				  << " \t L H" << std::endl
				  << std::endl;
		exit(EXIT_FAILURE);
	}

	// Computes sepia of 'input' and stores result in 'output'
	void matrixAddition(const std::vector<int> &inputA, const std::vector<int> &inputB, const uint width, const uint height, std::vector<int> &output)
	{
		std::cout << "Process on CPU (sequential)" << std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		for (uint i = 0; i < width * height; ++i)
		{
			output[i] = inputA[i] + inputB[i];
		}
		chrCPU.stop();
		std::cout << " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl
				  << std::endl;
	}

	// Compare two vectors
	bool compare(const std::vector<int> &a, const std::vector<int> &b)
	{
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			return false;
		}
		for (uint i = 0; i < a.size(); ++i)
		{
			// Floating precision can cause small difference between host and device
			if (std::abs(a[i] - b[i]) > 1)
			{
				std::cout << "Error at index " << i << ": a = " << uint(a[i]) << " - b = " << uint(b[i]) << std::endl;
				return false;
			}
		}
		return true;
	}

	// Main function
	void main(int argc, char **argv)
	{

		// Parse command line
		if (argc == 1)
		{
			std::cerr << "Please give a width L and a Height H..." << std::endl;
			printUsageAndExit(argv[0]);
		}

		int L = atoi(argv[1]);
		int H = atoi(argv[2]);

		// Create 2 output matrix
		std::vector<int> outputCPU(L * H);
		std::vector<int> outputGPU(L * H);

		std::vector<int> inputA(L * H);
		std::vector<int> inputB(L * H);

		for (int i = 0; i < L * H; i++)
		{
			inputA[i] = rand();
			inputB[i] = rand();
		}

		// Computation on CPU
		matrixAddition(inputA, inputB, L, H, outputCPU);

		std::cout
			<< "============================================" << std::endl
			<< "              STUDENT'S JOB !               " << std::endl
			<< "============================================" << std::endl;

		studentJob(inputA, inputB, L, H, outputGPU);

		std::cout << "============================================" << std::endl
				  << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compare(outputCPU, outputGPU))
		{
			std::cout << " -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
	}
}

int main(int argc, char **argv)
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
