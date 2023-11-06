#include "ImageProcessing.h"

#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
	cv::Mat image = cv::imread("standard_test_images\\lena_gray_512.tif", cv::IMREAD_GRAYSCALE);;
	unsigned char* newImageData = new unsigned char[image.cols * image.rows];

	cppImageProcessing::SobelBlur(image.data, newImageData, image.cols, image.rows);

	cv::Mat newImage(image.rows, image.cols, CV_8U, newImageData);

	cv::hconcat(image, newImage, image);
	cv::imshow("Image", image);
	cv::waitKey(0);
	
	return 0;
}