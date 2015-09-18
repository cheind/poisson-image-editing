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
#pragma warning (push)
#pragma warning (disable: 4244)
#include <Eigen/Sparse>
#include <Eigen/Dense>
#pragma warning (pop)

void solvePoissonEquations(cv::Mat_<uchar> bg, cv::Mat_<uchar> fg, cv::Mat_<uchar> fgm, cv::Mat_<float> vx, cv::Mat_<float> vy, cv::Mat &dst)
{
    const int width = bg.size().width;
    const int height = bg.size().height;
    cv::Rect bounds(0, 0, bg.cols, bg.rows);

    cv::Mat_<int> pixelToIndex(bg.size());
    int npixel = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            pixelToIndex(y, x) = fgm(y, x) ? npixel++ : -1;
        }
    }

    cv::Mat_<float> vxx, vyy;
    cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    cv::filter2D(vx, vxx, CV_32F, kernelx);
    cv::filter2D(vy, vyy, CV_32F, kernely);

    /*
    cv::Sobel(vx, vxx, CV_32F, 1, 0);
    cv::Sobel(vy, vyy, CV_32F, 0, 1);
    vxx *= 1.f / 8.f;
    vyy *= 1.f / 8.f;
    */

    std::vector<Eigen::Triplet<float> > triplets;

    triplets.reserve(5 * npixel); // maximum of five elements per pixel

    Eigen::VectorXf b(npixel);
    b.setZero();

    const int neighbors[4][2] = { {0, -1}, {0, 1},{ -1, 0},{ 1, 0 } };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            const cv::Point p = cv::Point(x, y);
            const int pid = pixelToIndex(p);
            
            if (fgm(p)) {
                int nneighbors = 0;
                for (int n = 0; n < 4; ++n) {
                    cv::Point q = cv::Point(x + neighbors[n][0], y + neighbors[n][1]);
                    if (bounds.contains(q)) {
                        ++nneighbors;
                        if (fgm(q)) {
                            const int qid = pixelToIndex(q);
                            triplets.push_back(Eigen::Triplet<float>(pid, qid, 1.f));
                        }
                        else {
                            b(pid) -= bg(q);
                        }
                    }                    
                }
                b(pid) += vxx(p) + vyy(p);
                triplets.push_back(Eigen::Triplet<float>(pid, pid, -(float)nneighbors));
            }
        }
    }

    Eigen::SparseMatrix<float> A(npixel, npixel);
    A.setFromTriplets(triplets.begin(), triplets.end());
    
    Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    Eigen::VectorXf result(npixel);
    result = solver.solve(b);

    std::cout << result(pixelToIndex(height / 2,  width /2)) << std::endl;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (fgm(y, x)) {
                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(result(pixelToIndex(y, x)));
            }
        }
    }
}

void seamlessImageCloning(cv::InputArray background_, cv::InputArray foreground_, cv::InputArray foregroundMask_, int offsetX, int offsetY, cv::OutputArray destination_)
{
    cv::Mat bg = background_.getMat();
    cv::Mat fg = foreground_.getMat();
    cv::Mat fgm = foregroundMask_.getMat();

    destination_.create(bg.size(), bg.type());
    cv::Mat dst = destination_.getMat();
    bg.copyTo(dst);
    
    cv::Rect overlapAreaBg = cv::Rect(0, 0, bg.cols, bg.rows) & cv::Rect(offsetX, offsetY, fg.cols, fg.rows);
    cv::Rect overlapAreaFg = cv::Rect(0, 0, std::min<int>(overlapAreaBg.width, fg.cols), std::min<int>(overlapAreaBg.height, fg.rows));

    cv::Mat_<float> vxf, vyf, vxb, vyb;

    cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    cv::filter2D(fg, vxf, CV_32F, kernelx);
    cv::filter2D(fg, vyf, CV_32F, kernely);
    cv::filter2D(bg, vxb, CV_32F, kernelx);
    cv::filter2D(bg, vyb, CV_32F, kernely);

    cv::Mat_<float> vx(overlapAreaFg.size()), vy(overlapAreaFg.size());
    cv::addWeighted(vxf(overlapAreaFg), 0.5f, vxb(overlapAreaBg), 0.5f, 0, vx);
    cv::addWeighted(vyf(overlapAreaFg), 0.5f, vyb(overlapAreaBg), 0.5f, 0, vy);

    /*
    cv::Sobel(fg, vx, CV_32F, 1, 0);
    cv::Sobel(fg, vy, CV_32F, 0, 1);

    vx *= 1.f / 8.f;
    vy *= 1.f / 8.f;
    */


    solvePoissonEquations(bg(overlapAreaBg), fg(overlapAreaFg), fgm(overlapAreaFg), vx(overlapAreaFg), vy(overlapAreaFg), dst(overlapAreaBg));
}

/** Main entry point */
int main(int argc, char **argv)
{
	if (argc != 5) {
		std::cerr << argv[0] << " background foreground offsetx offsety" << std::endl;
		return -1;
	}

    cv::Mat background = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat foreground = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    int offsetx = atoi(argv[3]);
    int offsety = atoi(argv[4]);

    cv::Mat mask(foreground.size(), CV_8UC1);
    mask.setTo(0);
    mask(cv::Rect(1, 1, mask.cols - 2, mask.rows - 2)).setTo(255);

    cv::Mat result;
    seamlessImageCloning(background, foreground, mask, offsetx, offsety, result);


    cv::imshow("background", background);
    cv::imshow("foreground", foreground);
    cv::imshow("output", result);
	cv::waitKey();

	return 0;
}




