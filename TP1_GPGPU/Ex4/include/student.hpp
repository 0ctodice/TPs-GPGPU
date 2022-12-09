/*
 * TP 1 - Premiers pas en CUDA
 * --------------------------
 * Ex 3: Filtre d'images sepia
 *
 * File: student.hpp
 * Author: Maxime MARIA
 */

#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"

namespace IMAC
{
	// Kernel:
	/// TODO

	// - input: input image RGB
	// - output: output image RGB
	void studentJob(const std::vector<int> &inputA, const std::vector<int> &inputB, const uint width, const uint height, std::vector<int> &output);

}

#endif
