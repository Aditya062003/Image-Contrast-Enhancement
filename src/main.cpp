#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

using std::chrono::high_resolution_clock;

void MyTimeOutput(const std::string& str, const high_resolution_clock::time_point& start_time, const high_resolution_clock::time_point& end_time)
{
    std::cout << str << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0 << "ms" << std::endl;
    return;
}

int main(int argc, char** argv)
{
    cv::Mat src = cv::imread(argv[1], 1);

    if (src.empty()) {
        std::cout << "Can't read image file." << std::endl;
        return -1;
    }

    // Create output directory if it doesn't exist
    system("mkdir -p output_images");

    high_resolution_clock::time_point start_time, end_time;

    start_time = high_resolution_clock::now();
    cv::Mat AINDANE_dst;
    AINDANE(src, AINDANE_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("AINDANE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat WTHE_dst;
    WTHE(src, WTHE_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("WTHE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat GCEHistMod_dst;
    GCEHistMod(src, GCEHistMod_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("GCEHistMod处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat LDR_dst;
    LDR(src, LDR_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("LDR处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat AGCWD_dst;
    AGCWD(src, AGCWD_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("AGCWD处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat AGCIE_dst;
    AGCIE(src, AGCIE_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("AGCIE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat IAGCWD_dst;
    IAGCWD(src, IAGCWD_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("IAGCWD处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat Ying_dst;
    Ying_2017_CAIP(src, Ying_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("Ying处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat CEusingLuminanceAdaptation_dst;
    CEusingLuminanceAdaptation(src, CEusingLuminanceAdaptation_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("CEusingLuminanceAdaptation处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat adaptiveImageEnhancement_dst;
    adaptiveImageEnhancement(src, adaptiveImageEnhancement_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("adaptiveImageEnhancement处理时间: ", start_time, end_time);
    
    start_time = high_resolution_clock::now();
    cv::Mat JHE_dst;
    JHE(src, JHE_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("JHE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat SEF_dst;
    SEF(src, SEF_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("SEF处理时间: ", start_time, end_time);

    // Save all images to output directory
    cv::imwrite("output_images/src.jpg", src);
    cv::imwrite("output_images/AINDANE_dst.jpg", AINDANE_dst);
    cv::imwrite("output_images/WTHE_dst.jpg", WTHE_dst);
    cv::imwrite("output_images/GCEHistMod_dst.jpg", GCEHistMod_dst);
    cv::imwrite("output_images/LDR_dst.jpg", LDR_dst);
    cv::imwrite("output_images/AGCWD_dst.jpg", AGCWD_dst);
    cv::imwrite("output_images/AGCIE_dst.jpg", AGCIE_dst);
    cv::imwrite("output_images/IAGCWD_dst.jpg", IAGCWD_dst);
    cv::imwrite("output_images/Ying_dst.jpg", Ying_dst);
    cv::imwrite("output_images/CEusingLuminanceAdaptation_dst.jpg", CEusingLuminanceAdaptation_dst);
    cv::imwrite("output_images/adaptiveImageEnhancement_dst.jpg", adaptiveImageEnhancement_dst);
    cv::imwrite("output_images/JHE_dst.jpg", JHE_dst);
    cv::imwrite("output_images/SEF_dst.jpg", SEF_dst);

    std::cout << "All enhanced images have been saved to the 'output_images' directory." << std::endl;
    
    return 0;
}