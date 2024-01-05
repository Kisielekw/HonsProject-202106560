#pragma once
#include <chrono>

static class cppImageProcessing
{
public:
	/// <summary>
	/// A Gaussian blure implementation in C++
	/// </summary>
	/// <param name="imageIn">The image being blured</param>
	/// <param name="imageOut">The output image of the Gaussian blure</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <param name="sd">The standard deviation of the Gaussian</param>
	/// <param name="kernalSize">The size of the kernal</param>
	/// <returns>The time it takes to run the algorithm in microseconds</returns>
	static std::chrono::microseconds GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sd, const unsigned int kernalSize);
	/// <summary>
	/// A Sobel edge detection implementation in C++
	/// </summary>
	/// <param name="imageIn">The image being processed by the Sobel opperator</param>
	/// <param name="imageOut">The image output by the Sobel opperator</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <returns>The time it takes to run the algorithm in microseconds</returns>
	static std::chrono::microseconds Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height);
	/// <summary>
	/// A K-Means implementation on a grayscale image in C++
	/// </summary>
	/// <param name="imageIn">The image geing segmented</param>
	/// <param name="imageOut">The output image of the K-Mean algorithm</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <param name="k">The number of clusters</param>
	/// <param name="iterationNum">The number of itterations to be preformed</param>
	/// <returns>The time it takes to run the algorithm in microseconds</returns>
	static std::chrono::microseconds KMeansGrayscale(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const int k, const int iterationNum);

private:
	static int Convolution(const unsigned char*, const float*, const int, const int, const int, const int, const unsigned int);
	static float GaussianFunction2D(const int, const int, const float);
};

static class cudaImageProcessing
{
public:
	/// <summary>
	/// A Gaussian blure implementation in CUDA
	/// </summary>
	/// <param name="imageIn">The image being blured</param>
	/// <param name="imageOut">The output image of the Gaussian blure</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <param name="sd">The standard deviation of the Gaussian</param>
	/// <param name="kernalSize">The size of the kernal</param>
	/// <returns>The time it takes to run the algorithm in microseconds</returns>
	static std::chrono::microseconds GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sd, const unsigned int kernalSize);
	/// <summary>
	/// A Sobel edge detection implementation in CUDA
	/// </summary>
	/// <param name="imageIn">The image being processed by the Sobel opperator</param>
	/// <param name="imageOut">The image output by the Sobel opperator</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <returns>The time it takes to run the algorithm in microseconds</returns>
	static std::chrono::microseconds Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height);
	/// <summary>
	/// A K-Means implementation on a grayscale image in CUDA
	/// </summary>
	/// <param name="imageIn">The image geing segmented</param>
	/// <param name="imageOut">The output image of the K-Mean algorithm</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <param name="k">The number of clusters</param>
	/// <param name="iterationNum">The number of itterations to be preformed</param>
	/// <returns>The time it takes to run the algorithm in microseconds</returns>
	static std::chrono::microseconds KMeansGrayscale(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const int k, const int iterationNum);
};