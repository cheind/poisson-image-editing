/**
 This file is part of PoissonBlend.
 
 Copyright Christoph Heindl 2015
 
 PoissonBlend is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 PoissonBlend is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with PoissonBlend.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <blend/poisson_blend.h>
#include <opencv2/opencv.hpp>

/**
 
 Naive image cloning by just copying the values from foreground over background
 
 */
void naiveClone(cv::InputArray background_,
                cv::InputArray foreground_,
                cv::InputArray foregroundMask_,
                int offsetX, int offsetY,
                cv::OutputArray destination_)
{
    cv::Mat bg = background_.getMat();
    cv::Mat fg = foreground_.getMat();
    cv::Mat fgm = foregroundMask_.getMat();
    
    destination_.create(bg.size(), bg.type());
    cv::Mat dst = destination_.getMat();
    
    cv::Rect overlapAreaBg, overlapAreaFg;
    blend::detail::findOverlap(background_, foreground_, offsetX, offsetY, overlapAreaBg, overlapAreaFg);
    
    bg.copyTo(dst);
    fg(overlapAreaFg).copyTo(dst(overlapAreaBg), fgm(overlapAreaFg));
    
}

/**
 
 Main entry point.
 
 */
int main(int argc, char **argv)
{
    if (argc != 6) {
        std::cerr << argv[0] << " background foreground mask offsetx offsety" << std::endl;
        return -1;
    }
    
    cv::Mat background = cv::imread(argv[1]);
    cv::Mat foreground = cv::imread(argv[2]);
    cv::Mat mask = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
    int offsetx = atoi(argv[4]);
    int offsety = atoi(argv[5]);
    
    
    cv::Mat result;
    
    naiveClone(background, foreground, mask, offsetx, offsety, result);
    cv::imshow("Naive", result);
    cv::imwrite("naive.png", result);
    
    blend::seamlessClone(background, foreground, mask, offsetx, offsety, result, blend::CLONE_MIXED_GRADIENTS);
    cv::imshow("Mixed Gradients", result);
    cv::imwrite("mixed-gradients.png", result);
    
    blend::seamlessClone(background, foreground, mask, offsetx, offsety, result, blend::CLONE_FOREGROUND_GRADIENTS);
    cv::imshow("Foreground Gradients", result);
    cv::imwrite("foreground-gradients.png", result);
    
    blend::seamlessClone(background, foreground, mask, offsetx, offsety, result, blend::CLONE_AVERAGED_GRADIENTS);
    cv::imshow("Averaged Gradients", result);
    cv::imwrite("averaged-gradients.png", result);
    
    cv::waitKey();
    
    return 0;
}




