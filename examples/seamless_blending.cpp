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

#include <blend/blend.h>
#include <opencv2/opencv.hpp>


/**
 
 Naive image blending by just copying the values from foreground over background
 
 */
void naiveBlend(cv::InputArray first_,
                cv::InputArray second_,
                cv::InputArray mask_,
                cv::OutputArray dst_)
{
    
    first_.getMat().copyTo(dst_);
    second_.getMat().copyTo(dst_, (255 - mask_.getMat()));
}

/**
 
 Main entry point.
 
 */
int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << argv[0] << " first second mask" << std::endl;
        return -1;
    }
    
    cv::Mat first = cv::imread(argv[1]);
    cv::Mat second = cv::imread(argv[2]);
    cv::Mat mask = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
    
    
    cv::Mat result;
    
    naiveBlend(first, second, mask, result);
    cv::imshow("Naive Blend", result);
    cv::imwrite("naive-blend.png", result);
    
    
    blend::seamlessBlend(first, second, mask, result);
    cv::imshow("Seamless Blend", result);
    cv::imwrite("seamless-blend.png", result);
    
    cv::waitKey();
    
    return 0;
}




