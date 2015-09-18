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

void solvePoissonEquations(cv::Mat_<uchar> bg, cv::Mat_<uchar> fg, cv::Mat_<uchar> fgm, cv::Mat_<float> vx, cv::Mat_<float> vy, cv::Mat dst)
{
    const int width = bg.size().width;
    const int height = bg.size().height;
    cv::Rect bounds(0, 0, bg.cols, bg.rows);

    // Build a mapping from masked pixel to linear index.
    cv::Mat_<int> pixelToIndex(bg.size());
    int npixel = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            pixelToIndex(y, x) = fgm(y, x) ? npixel++ : -1;
        }
    }

    // Divergence of guidance field
    cv::Mat_<float> vxx, vyy;
    cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    cv::filter2D(vx, vxx, CV_32F, kernelx);
    cv::filter2D(vy, vyy, CV_32F, kernely);

    // Sparse matrix A is being build with row, column, value triplets
    std::vector<Eigen::Triplet<float> > triplets;
    triplets.reserve(5 * npixel); // Maximum of five elements per pixel

    Eigen::VectorXf b(npixel);
    b.setZero();

    const int neighbors[4][2] = { {0, -1}, {0, 1},{ -1, 0},{ 1, 0 }};
    

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
    
    Eigen::SparseLU< Eigen::SparseMatrix<float> > solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    Eigen::VectorXf result(npixel);
    result = solver.solve(b);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (fgm(y, x)) {
                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(result(pixelToIndex(y, x)));
            }
        }
    }
}

struct ForegroundGradientField {
    
    void computeGuidanceVectorField(cv::Mat background, cv::Mat foreground, cv::Mat &vx, cv::Mat &vy)
    {
        cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
        cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
        
        cv::filter2D(foreground, vx, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(foreground, vy, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    }
};

struct WeightedAveragedGradientField {
    
    WeightedAveragedGradientField(float wForeground)
    : weightForeground(wForeground)
    {}
    
    void computeGuidanceVectorField(cv::Mat background, cv::Mat foreground, cv::Mat &vx, cv::Mat &vy)
    {
        cv::Mat_<float> vxf, vyf, vxb, vyb;
        cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
        cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
        
        cv::filter2D(foreground, vxf, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(foreground, vyf, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(background, vxb, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(background, vyb, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        
        cv::addWeighted(vxf, weightForeground, vxb, 1.f - weightForeground, 0, vx);
        cv::addWeighted(vyf, weightForeground, vyb, 1.f - weightForeground, 0, vy);
    }
    
    float weightForeground;
};

struct MixedGradientField {
    
    void computeGuidanceVectorField(cv::Mat background, cv::Mat foreground, cv::Mat &vx, cv::Mat &vy)
    {
        cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
        cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
        
        cv::Mat_<float> vxf, vyf, vxb, vyb;
        cv::filter2D(foreground, vxf, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(foreground, vyf, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(background, vxb, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(background, vyb, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        
        vx.create(vxf.size(), vxf.type());
        vy.create(vyf.size(), vyf.type());
        
        for(int id = 0; id < (vx.rows * vx.cols); ++id)
        {
            cv::Vec2f g[2] = {cv::Vec2f(vxf.ptr<float>()[id], vyf.ptr<float>()[id]),
                              cv::Vec2f(vxb.ptr<float>()[id], vyb.ptr<float>()[id])};
            
            int which = (g[0].dot(g[0]) > g[1].dot(g[1])) ? 0 : 1;
            
            vx.ptr<float>()[id] = g[which][0];
            vy.ptr<float>()[id] = g[which][1];
        }
        
    }
};

template<class GuidanceFieldGenerator>
void seamlessImageCloning(cv::InputArray background_,
                          cv::InputArray foreground_,
                          cv::InputArray foregroundMask_,
                          int offsetX, int offsetY,
                          GuidanceFieldGenerator gen,
                          cv::OutputArray destination_)
{
    cv::Mat bg = background_.getMat();
    cv::Mat fg = foreground_.getMat();
    cv::Mat fgm = foregroundMask_.getMat();

    destination_.create(bg.size(), bg.type());
    cv::Mat dst = destination_.getMat();
    bg.copyTo(dst);
    
    cv::Rect overlapAreaBg = cv::Rect(0, 0, bg.cols, bg.rows) & cv::Rect(offsetX, offsetY, fg.cols, fg.rows);
    cv::Rect overlapAreaFg = cv::Rect(0, 0, std::min<int>(overlapAreaBg.width, fg.cols), std::min<int>(overlapAreaBg.height, fg.rows));
    
    std::vector<cv::Mat> bgChannels, fgChannels, dstChannels;
    
    cv::split(fg, fgChannels);
    cv::split(bg, bgChannels);
    cv::split(dst, dstChannels);
    
    cv::Mat_<float> vx(overlapAreaFg.size()), vy(overlapAreaFg.size());
    
    for (int c = 0; c < bg.channels(); ++c) {
        
        gen.computeGuidanceVectorField(bgChannels[c](overlapAreaBg),
                                       fgChannels[c](overlapAreaFg),
                                       vx, vy);

        solvePoissonEquations(bgChannels[c](overlapAreaBg),
                              fgChannels[c](overlapAreaFg),
                              fgm(overlapAreaFg), vx, vy,
                              dstChannels[c](overlapAreaBg));
    }
    
    cv::merge(dstChannels, dst);
}

/** Main entry point */
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

    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
    cv::rectangle(mask,cv::Rect(0, 0, mask.cols - 1, mask.rows - 1), 0, 1);
    
    cv::Mat result;
    
    seamlessImageCloning(background, foreground, mask, offsetx, offsety, ForegroundGradientField(), result);
    cv::imshow("ForegroundGradientField", result);
    
    seamlessImageCloning(background, foreground, mask, offsetx, offsety, WeightedAveragedGradientField(0.5f), result);
    cv::imshow("WeightedAveragedGradientField", result);
    
    seamlessImageCloning(background, foreground, mask, offsetx, offsety, MixedGradientField(), result);
    cv::imshow("MixedGradientField", result);
    
    cv::imshow("background", background);
	cv::waitKey();

	return 0;
}




