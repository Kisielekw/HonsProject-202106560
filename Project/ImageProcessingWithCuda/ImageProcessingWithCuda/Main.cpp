#include "ImageProcessing.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

enum Algorithms {
	Sobel,
	KMeans
};

std::string* GetImageFile()
{
	std::string directory_path = "Standard_Test_Images\\";
	std::vector<std::string> fileNames;
	
	int count = 1;
	for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory_path))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".tif")
		{
			std::cout << count++ << ". " << entry.path().string() << std::endl;
			fileNames.push_back(entry.path().string());
		}
	}

	int input = 0;
	std::cin >> input;
	return new std::string(fileNames[input - 1]);
}

Algorithms GetAlgorithm()
{
	int userInput = 0;
	do
	{
		std::cout << "Select option:" << std::endl
			<< "1. Sobel" << std::endl
			<< "2. K-Means" << std::endl;
		std::cin >> userInput;
	} while (userInput < 1 || userInput > 2);

	return static_cast<Algorithms>(userInput - 1);
}

void SobelBenchmark(const std::string& testFile)
{
	//User inputs for the benchmark test
	int startSize = 0;
	int endSize = 0;
	int iterationNum = 0;

	std::cout << "Enter the starting image size: ";
	std::cin >> startSize;
	std::cout << "Enter the ending image size: ";
	std::cin >> endSize;
	std::cout << "Enter the number of iterations: ";
	std::cin >> iterationNum;

	std::ofstream output(testFile + ".csv");

	output << "Image Size,CPU Time,CUDA Time" << std::endl;

	for (int size = startSize; size <= endSize; size++)
	{
		std::cout << '\r' << "percentage complete: " << (size - startSize + 1) * 100 / (endSize - startSize + 1) << "%";

		output << (size * size) << ",";

		cv::Mat image = cv::Mat(size, size, CV_8UC1);

		unsigned char* cppImageData = new unsigned char[image.cols * image.rows];
		unsigned char* cudaImageData = new unsigned char[image.cols * image.rows];

		long cppDurationSum = 0;
		long cudaDurationSum = 0;

		for (int i = 0; i < iterationNum; i++)
		{
			std::chrono::microseconds cppDuration = cppImageProcessing::Sobel(image.data, cppImageData, image.cols, image.rows);
			std::chrono::microseconds cudaDuration = cudaImageProcessing::Sobel(image.data, cudaImageData, image.cols, image.rows);

			cppDurationSum += long(cppDuration.count());
			cudaDurationSum += long(cudaDuration.count());
		}

		output << cppDurationSum / iterationNum << "," << cudaDurationSum / iterationNum << std::endl;

		delete[] cudaImageData;
		delete[] cppImageData;
	}

	output.close();

	std::cout << std::endl << "Benchmark finished" << std::endl;
}

void KMeansBenchmark(const std::string& testFile)
{
	int benchmarkIterationNum = 0;
	int imageSizeStart = 0;
	int imageSizeEnd = 0;
	int kStart = 0;
	int kEnd = 0;
	int iterationNumStart = 0;
	int iterationNumEnd = 0;

	std::cout << "Enter the number of iterations to be preformed for each benchmark: ";
	std::cin >> benchmarkIterationNum;
	std::cout << "Enter the starting image size: ";
	std::cin >> imageSizeStart;
	std::cout << "Enter the ending image size: ";
	std::cin >> imageSizeEnd;
	std::cout << "Enter the starting number of clusters: ";
	std::cin >> kStart;
	std::cout << "Enter the ending number of clusters: ";
	std::cin >> kEnd;
	std::cout << "Enter the starting number of iterations in the algorithm: ";
	std::cin >> iterationNumStart;
	std::cout << "Enter the ending number of iterations in the algorithm: ";
	std::cin >> iterationNumEnd;

	std::ofstream output(testFile + ".csv");

	output << "Image Size,Number of Clusters,Number of Iterations,CPU Time,CUDA Time" << std::endl;

	int count = 0;

	for (int imageSize = imageSizeStart; imageSize <= imageSizeEnd; imageSize++)
	{
		cv::Mat image = cv::Mat(imageSize, imageSize, CV_8UC1);

		cv::randn(image, 128, 20);

		unsigned char* cppImageData = new unsigned char[image.cols * image.rows];
		unsigned char* cudaImageData = new unsigned char[image.cols * image.rows];

		for (int k = kStart; k <= kEnd; k++)
		{
			for (int iterationNum = iterationNumStart; iterationNum <= iterationNumEnd; iterationNum++)
			{
				count++;
				std::cout << '\r' << "percentage complete: " << (count * 100) / ((imageSizeEnd - imageSizeStart + 1) * (kEnd - kStart + 1) * (iterationNumEnd - iterationNumStart + 1)) << "%";

				output << (imageSize * imageSize) << "," << k << "," << iterationNum << ",";

				long cppDurationSum = 0;
				long cudaDurationSum = 0;

				for (int i = 0; i < benchmarkIterationNum; i++)
				{
					std::chrono::microseconds cppDuration = cppImageProcessing::KMeansGrayscale(image.data, cppImageData, image.cols, image.rows, k, iterationNum);
					std::chrono::microseconds cudaDuration = cudaImageProcessing::KMeansGrayscale(image.data, cudaImageData, image.cols, image.rows, k, iterationNum);

					cppDurationSum += long(cppDuration.count());
					cudaDurationSum += long(cudaDuration.count());
				}

				output << cppDurationSum / benchmarkIterationNum << "," << cudaDurationSum / benchmarkIterationNum << std::endl;
			}
		}

		delete[] cudaImageData;
		delete[] cppImageData;
	}

	std::cout << std::endl << "Benchmark finished" << std::endl;
}

void SaveImage(cv::Mat& image)
{
	std::cout << "Save image? (y/n): ";
	char input;
	std::cin >> input;

	if (input == 'y')
	{
		std::string saveFile;
		std::cout << "Enter name of file to save image to with file extension: ";
		std::cin >> saveFile;

		cv::imwrite(saveFile, image);
	}
}

void SobelDisplay(const std::string& imageFile)
{
	cv::Mat image = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
	cv::Mat cppImage = cv::Mat(image.rows, image.cols, CV_8UC1);
	cv::Mat cudaImage = cv::Mat(image.rows, image.cols, CV_8UC1);

	cppImageProcessing::Sobel(image.data, cppImage.data, image.cols, image.rows);
	cudaImageProcessing::Sobel(image.data, cudaImage.data, image.cols, image.rows);

	cv::hconcat(image, cppImage, image);
	cv::hconcat(image, cudaImage, image);

	cv::imshow("Sobel", image);
	cv::waitKey(0);
	if(cv::getWindowProperty("Sobel", cv::WND_PROP_VISIBLE))
		cv::destroyWindow("Sobel");

	SaveImage(image);
}

void KMeansDisplay(const std::string& imageFile)
{
	int k = 0;
	int iterationNum = 0;

	std::cout << "Enter the number of clusters: ";
	std::cin >> k;
	std::cout << "Enter the number of iterations: ";
	std::cin >> iterationNum;

	cv::Mat image = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
	cv::Mat cppImage = cv::Mat(image.rows, image.cols, CV_8UC1);
	cv::Mat cudaImage = cv::Mat(image.rows, image.cols, CV_8UC1);

	cppImageProcessing::KMeansGrayscale(image.data, cppImage.data, image.cols, image.rows, k, iterationNum);
	cudaImageProcessing::KMeansGrayscale(image.data, cudaImage.data, image.cols, image.rows, k, iterationNum);

	cv::hconcat(image, cppImage, image);
	cv::hconcat(image, cudaImage, image);

	cv::imshow("K-Means", image);
	cv::waitKey(0);
	if (cv::getWindowProperty("K-Means", cv::WND_PROP_VISIBLE))
		cv::destroyWindow("K-Means");

	SaveImage(image);
}

int main()
{
	std::cout << "Image processing with CUDA" << std::endl;
	
	int userInput = 0;
	do
	{
		std::cout << "Select option:" << std::endl 
			<< "1. Benchmark" << std::endl 
			<< "2. Display Result" << std::endl
			<< "3. Exit" << std::endl;
		std::cin >> userInput;

		if (userInput == 3)
			continue;

		Algorithms algorithm = GetAlgorithm();

		if (userInput == 1)
		{
			std::string testFile;
			std::cout << "Enter name of file to save benchmark results to: ";
			std::cin >> testFile;

			if (algorithm == 0)
				SobelBenchmark(testFile);
			else if (algorithm == 1)
				KMeansBenchmark(testFile);
		}
		else if (userInput == 2)
		{
			std::string* imageFile = GetImageFile();
			if (algorithm == 0)
				SobelDisplay(*imageFile);
			else if (algorithm == 1)
				KMeansDisplay(*imageFile);

			delete imageFile;
		}

	} while (userInput != 3);

	return 0;
}

int Main()
{
	cv::Mat imageIn = cv::imread("Standard_Test_Images\\image.png", cv::IMREAD_GRAYSCALE);
	cv::Mat imageOut = cv::Mat(imageIn.rows, imageIn.cols, CV_16UC1);

	cv::hconcat(imageIn, imageOut, imageIn);

	cv::imwrite("Standard_Test_Images\\imageOutCanny.png", imageIn);

	return 1;
}