/**
This file is part of Poisson Image Editing.

Copyright Christoph Heindl 2015

Poisson Image Editing is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Poisson Image Editing is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Poisson Image Editing.  If not, see <http://www.gnu.org/licenses/>.
*/

#define CATCH_CONFIG_MAIN  
#include "catch.hpp"
#include "test_config.h"
#include <blend/poisson_solver.h>
#include <opencv2/opencv.hpp>

void showResult(cv::Mat initial, cv::Mat result)
{
    initial = initial.clone();
    result = result.clone();

    initial.convertTo(initial, CV_8UC1);
    result.convertTo(result, CV_8UC1);

    float ratio = initial.rows / (float)initial.cols;
    
    cv::resize(initial, initial, cv::Size(400, static_cast<int>(400.f * ratio)), 0, 0, CV_INTER_NN);
    cv::resize(result, result, cv::Size(400, static_cast<int>(400.f * ratio)), 0, 0, CV_INTER_NN);

    cv::imshow("input", initial);
    cv::imshow("output", result);
    cv::waitKey();
}

TEST_CASE("Laplacian-1d")
{
    const int width = 40;
    const int height = 1;

    cv::Mat f(height, width, CV_32FC1);
    f.setTo(0);

    cv::Mat bmask(height, width, CV_8UC1);
    bmask.setTo(blend::constants::UNKNOWN);
    bmask.at<uchar>(0, 0) = blend::constants::DIRICHLET_BD;
    bmask.at<uchar>(0, width - 1) = blend::constants::DIRICHLET_BD;

    cv::Mat bvalues(height, width, CV_32FC1);
    bvalues.setTo(0);
    bvalues.at<float>(0, 0) = 0;
    bvalues.at<float>(0, width - 1) = 255;

    cv::Mat result;
    blend::solvePoissonEquations(f, bmask, bvalues, result);

#ifdef BLEND_TESTS_VERBOSE
    showResult(bvalues, result);    
#endif
    
    // Result should be a gradient
    float offset = 255.f / (width - 1);
    for (int i = 0; i < width; ++i) {
        REQUIRE(result.at<float>(0, i) == Approx(i*offset).epsilon(0.01));
    }
}

TEST_CASE("Laplacian-2d")
{
    const int width = 40;
    const int height = 40;

    cv::Mat f(height, width, CV_32FC1);
    f.setTo(0);

    cv::Mat bmask(height, width, CV_8UC1);
    bmask.setTo(blend::constants::UNKNOWN);
    bmask.at<uchar>(0, 0) = blend::constants::DIRICHLET_BD;
    bmask.at<uchar>(height - 1, width - 1) = blend::constants::DIRICHLET_BD;
        
    cv::Mat bvalues(height, width, CV_32FC1);
    bvalues.setTo(0);
    bvalues.at<float>(0, 0) = 255;
    bvalues.at<float>(height - 1, width - 1) = 0;
    
    cv::Mat result;
    blend::solvePoissonEquations(f, bmask, bvalues, result);

    // Result should be a gradient
    /*float offset = 255.f / (width - 1);
    for (int i = 0; i < width; ++i) {
        REQUIRE(result.at<float>(0, i) == Approx(i*offset).epsilon(0.01));
    }

    cv::Mat lap, abs_lap;
    cv::Laplacian(result, lap, CV_32F, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(lap, abs_lap);
    */

    

#ifdef BLEND_TESTS_VERBOSE
    showResult(bvalues, result);
#endif
}
