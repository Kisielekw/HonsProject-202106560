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
	/// <returns></returns>
	static std::chrono::microseconds GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sd, const unsigned int kernalSize);
	/// <summary>
	/// A Sobel edge detection implementation in C++
	/// </summary>
	/// <param name="imageIn">The image being processed by the Sobel opperator</param>
	/// <param name="imageOut">The image output by the Sobel opperator</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <returns></returns>
	static std::chrono::microseconds Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height);
	/// <summary>
	/// A Difference of Gaussian edge detection implementation in C++
	/// </summary>
	/// <param name="imageIn">The image being blured</param>
	/// <param name="imageOut">The output image of the Gaussian blure</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <param name="sd1">The standard deviation of the first Gaussian blure</param>
	/// <param name="sd2">The standard deviation of the second Gaussian blure</param>
	/// <param name="kernalSize">The size of the kernal</param>
	/// <returns></returns>
	static std::chrono::microseconds DoG(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sd1, const float sd2, const unsigned int kernalSize);
	/// <summary>
	/// A thresholding implementation in C++
	/// </summary>
	/// <param name="imageIn">The image being thresholded</param>
	/// <param name="imageOut">The output image of the threshold</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <param name="threshold">The threshold</param>
	/// <returns></returns>
	static std::chrono::microseconds Threshold(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const unsigned char threshold);
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
	/// <returns></returns>
	static std::chrono::microseconds GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sd, const unsigned int kernalSize);
	/// <summary>
	/// A Sobel edge detection implementation in CUDA
	/// </summary>
	/// <param name="imageIn">The image being processed by the Sobel opperator</param>
	/// <param name="imageOut">The image output by the Sobel opperator</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <returns></returns>
	static std::chrono::microseconds Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height);
	/// <summary>
	/// A Difference of Gaussian edge detection implementation in CUDA
	/// </summary>
	/// <param name="imageIn">The image being blured</param>
	/// <param name="imageOut">The output image of the Gaussian blure</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <param name="sd1">The standard deviation of the first Gaussian blure</param>
	/// <param name="sd2">The standard deviation of the second Gaussian blure</param>
	/// <param name="kernalSize">The size of the kernal</param>
	/// <returns></returns>
	static std::chrono::microseconds DoG(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sd1, const float sd2, const unsigned int kernalSize);
	/// <summary>
	/// A thresholding implementation in CUDA
	/// </summary>
	/// <param name="imageIn">The image being thresholded</param>
	/// <param name="imageOut">The output image of the threshold</param>
	/// <param name="width">The width of the image</param>
	/// <param name="height">The height of the image</param>
	/// <param name="threshold">The threshold</param>
	/// <returns></returns>
	static std::chrono::microseconds Threshold(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const unsigned char threshold);
};