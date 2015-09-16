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

#include <iostream>
#include <opencv2/opencv.hpp>

/** Main entry point */
int main(int argc, char **argv)
{
	if (argc != 5) {
		std::cerr << argv[0] << " background foreground offsetx offsety" << std::endl;
		return -1;
	}

    cv::Mat background = cv::imread(argv[1]);
    cv::Mat foreground = cv::imread(argv[2]);
    int offsetx = atoi(argv[3]);
    int offsety = atoi(argv[4]);

    cv::Mat mask(foreground.size(), CV_8UC1);
    mask.setTo(0);
    mask(cv::Rect(1, 1, mask.cols - 2, mask.rows - 2)).setTo(255);

    cv::Mat result;
    blend::seamlessImageCloning(background, foreground, mask, offsetx, offsety, result);


    cv::imshow("background", background);
    cv::imshow("foreground", foreground);
    cv::imshow("output", result);
	cv::waitKey();

	return 0;
}




