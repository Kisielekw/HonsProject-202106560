#include "ImageProcessing.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

void SobelBenchmark(const std::string &testFile, const int iterationNum, const int startSize, const int endSize)
{
	std::ofstream output(testFile + ".csv");

	output << "Image Size,CPU Time,CUDA Time" << std::endl;

	for (int size = startSize; size <= endSize; size++)
	{
		std::cout << '\r' << "percentage complete: " << (size - startSize) * 100 / (endSize - startSize) << "%";

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

			cppDurationSum += cppDuration.count();
			cudaDurationSum += cudaDuration.count();
		}

		output << cppDurationSum / iterationNum << "," << cudaDurationSum / iterationNum << std::endl;

		delete[] cudaImageData;
		delete[] cppImageData;
	}

	output.close();

	std::cout << std::endl << "Benchmark finished" << std::endl;
}

void DisplaySobelImage(const std::string &imagePath)
{
	cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

	cv::Mat cppImage = cv::Mat(image.rows, image.cols, CV_8UC1);
	cv::Mat cudaImage = cv::Mat(image.rows, image.cols, CV_8UC1);

	cppImageProcessing::Sobel(image.data, cppImage.data, image.cols, image.rows);
	cudaImageProcessing::Sobel(image.data, cudaImage.data, image.cols, image.rows);

	cv::hconcat(image, cppImage, image);
	cv::hconcat(image, cudaImage, image);

	cv::imshow("Sobel Comparison", image);

	cv::waitKey(0);
	if(cv::getWindowProperty("Sobel Comparison", cv::WND_PROP_VISIBLE) > 0)
		cv::destroyWindow("Sobel Comparison");
}

int main()
{
	std::cout << "Image Processing with CUDA" << std::endl;
	while (true)
	{
		std::cout << "1. Sobel Benchmark" << std::endl;
		std::cout << "2. Display Sobel Image" << std::endl;

		int input;
		std::cin >> input;

		if (input == 1)
		{
			std::cout << "Enter the number of iterations: ";
			int iterationNum;
			std::cin >> iterationNum;

			std::cout << "Enter the start size: ";
			int startSize;
			std::cin >> startSize;

			std::cout << "Enter the end size: ";
			int endSize;
			std::cin >> endSize;

			std::cout << "Enter the test file name: ";
			std::string testFile;
			std::cin >> testFile;

			SobelBenchmark(testFile, iterationNum, startSize, endSize);
			
			continue;
		}
		else if (input == 2)
		{
			std::string directory_path = "Standard_Test_Images\\";
			std::vector<std::string> file_names;
			int count = 1;
			for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
				if (entry.is_regular_file() && entry.path().extension() == ".tif") {
					std::cout << count++ << ". " << entry.path().string() << std::endl;
					file_names.push_back(entry.path().string());
				}
			}

			while (true)
			{
				std::cout << "Enter the image number: ";
				int imageNum;
				std::cin >> imageNum;

				if (imageNum < 1 || imageNum > file_names.size())
				{
					std::cout << "Invalid image number" << std::endl;
					continue;
				}

				DisplaySobelImage(file_names[imageNum - 1]);
				break;
			}

			continue;
		}

		break;
	}

	return 0;
}