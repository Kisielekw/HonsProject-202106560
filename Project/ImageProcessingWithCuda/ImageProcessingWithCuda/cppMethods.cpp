#include "ImageProcessing.h"
#include <cmath>
#include <algorithm>

/// <summary>
/// Preforms a Sobel Edge Detection on the imageIn and stores the result in imageOut.
/// </summary>
/// <param name="imageIn">The image that the Sobel is meant to be preformed on</param>
/// <param name="imageOut"The output image of the Sobel></param>
/// <param name="width">The width of the image</param>
/// <param name="height">the height of the image</param>
/// <returns>Returns the time it took to preform the filter in microseconds</returns>
std::chrono::microseconds cppImageProcessing::Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height)
{
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	float Gx[9] = {
		1, 0, -1,
		2, 0, -2,
		1, 0, -1
	};
	float Gy[9] = {
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1
	};
	
	for (int i = 0; i < 2; i++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int sobelX = Convolution(imageIn, Gx, x, y, width, height, 3);
				int sobelY = Convolution(imageIn, Gy, x, y, width, height, 3);

				int magnitude = static_cast<int>(sqrt(static_cast<double>((sobelX * sobelX) + (sobelY * sobelY))));

				imageOut[x + (y * width)] = static_cast<unsigned char>(std::min(255, std::max(0, magnitude)));
			}
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	return duration;
}

/// <summary>
/// Preforms a Gaussian Blur on the imageIn and stores the result in imageOut.
/// </summary>
/// <param name="imageIn">The image that the Gaussian Blure is meant to be preformed on</param>
/// <param name="imageOut">The output image of the Gaussian Blure</param>
/// <param name="width">The width of the image</param>
/// <param name="height">The height of the image</param>
/// <param name="sigma">The standard ceviation of the blure</param>
/// <returns>Returns the time it took to preform the filter in microseconds</returns>
std::chrono::microseconds cppImageProcessing::GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sigma, const unsigned int kernalSize)
{
	if(kernalSize % 2 != 1)
		throw std::invalid_argument("Kernal size must be odd");

	float* gaussKernel = new float[kernalSize * kernalSize];

	int halfKernalSize = (kernalSize - 1) / 2;

	for (int y = -halfKernalSize; y <= halfKernalSize; y++)
	{
		for (int x = -halfKernalSize; x <= halfKernalSize; x++)
		{
			gaussKernel[(x + halfKernalSize) + ((y + halfKernalSize) * kernalSize)] = GaussianFunction2D(x, y, sigma);
		}
	}

	float kernelSum = 0;
	for (int i = 0; i < kernalSize * kernalSize; i++) {
		kernelSum += gaussKernel[i];
	}

	for (int i = 0; i < kernalSize * kernalSize; i++) {
		gaussKernel[i] /= kernelSum;
	}

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			imageOut[x + y * width] = Convolution(imageIn, gaussKernel, x, y, width, height, kernalSize);
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	return duration;
}

float cppImageProcessing::GaussianFunction2D(const int x, const int y, const float sigma)
{
	double expPow = static_cast<double>((x * x) + (y * y)) / (2.0 * static_cast<double>(sigma * sigma));
	double exponentail = exp(-expPow);
	return exponentail / (2 * 3.1415 * (sigma * sigma));
}

int cppImageProcessing::Convolution(const unsigned char* imageIn, const float* kernal, const int x, const int y, const int width, const int height, const unsigned int kernalSize)
{
	int sum = 0;

	int halfKernalSize = (kernalSize - 1) / 2;

	for (int ky = -halfKernalSize; ky <= halfKernalSize; ky++)
	{
		int curentY = ky + y;
		if (curentY >= height || curentY < 0)
			continue;
		for (int kx = -halfKernalSize; kx <= halfKernalSize; kx++)
		{
			int curentX = kx + x;
			if (curentX >= width || curentX < 0)
				continue;
			sum += imageIn[curentX + (curentY * width)] * kernal[(kx + halfKernalSize) + ((ky + halfKernalSize) * kernalSize)];
		}
	}

	return sum;
}