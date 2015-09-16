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
#pragma warning (push)
#pragma warning (disable: 4244)
#include <Eigen/Sparse>
#include <Eigen/Dense>
#pragma warning (pop)

namespace blend {

    void computeGradientImage(cv::Mat src, cv::Mat &gradX, cv::Mat &gradY) 
    {
        cv::Mat blurred;
        cv::GaussianBlur(src, blurred, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

        cv::Sobel(blurred, gradX, CV_32F, 1, 0);
        cv::Sobel(blurred, gradY, CV_32F, 0, 1);
    }

    void computeGuidanceVectorField(cv::Mat_<uchar> fgm, cv::Mat_<float> bgGradX, cv::Mat_<float> bgGradY, cv::Mat_<float> fgGradX, cv::Mat_<float> fgGradY, cv::Mat_<cv::Vec2f> &v)
    {
        v.create(bgGradX.size());

        const int width = bgGradX.size().width;
        const int height = bgGradX.size().height;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (!fgm(y, x)) {
                    v(y, x) = cv::Vec2f(bgGradX(y, x), bgGradY(y, x));
                }
                else {
                    cv::Vec2f bgGrad(bgGradX(y, x), bgGradY(y, x));
                    cv::Vec2f fgGrad(fgGradX(y, x), fgGradY(y, x));
                    v(y, x) = (bgGrad.dot(bgGrad) > fgGrad.dot(fgGrad)) ? bgGrad : fgGrad;
                }
            }
        }
    }

    void solvePoissonEquations(cv::Mat_<uchar> bg, cv::Mat_<uchar> fg, cv::Mat_<uchar> fgm, cv::Mat_<cv::Vec2f> v, Eigen::VectorXf &x)
    {
        const int npixel = bg.size().area();
        const int width = bg.size().width;
        const int height = bg.size().height;

        Eigen::SparseMatrix<float> A(npixel, npixel);
        Eigen::VectorXf b(npixel);

        A.reserve(5 * height); // usually five elements per row
        A.setZero();
        b.setZero();

        cv::Rect bounds(0, 0, bg.cols, bg.rows);

        auto updatePQ = [&](cv::Point p, cv::Point q) {
            const int pid = p.y * width + p.x;
            const int qid = q.y * width + q.x;
            
            if (bounds.contains(q)) {

                A.coeffRef(pid, pid) += 1;
                if (fgm(q)) {
                    A.coeffRef(pid, qid) = -1.f;
                } else {
                    b(pid) += bg(q);
                }

                int dir = ((p - q).x == 0) ? 1 : 0;
                b(pid) += v(p)[dir] - v(q)[dir];
            }
        };

        for (int y = 0; y < height; ++y) {            
            for (int x = 0; x < width; ++x) {

                const cv::Point p = cv::Point(x, y);
                int id = y * width + x;
                
                if (!fgm(y, x)) {
                    // Outside of mask, solution should be the color of background.
                    
                    A.coeffRef(id, id) = 1.f;
                    b(id) = static_cast<float>(bg(y, x));
                } else {
                    // Inside of mask, build equation from neighbors
                    
                    cv::Point q = cv::Point(x, y - 1);
                    updatePQ(p, q);

                    q = cv::Point(x + 1, y);
                    updatePQ(p, q);

                    q = cv::Point(x, y + 1);
                    updatePQ(p, q);

                    q = cv::Point(x - 1, y);
                    updatePQ(p, q);
                }
            }
        }

        Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);

        x.resize(npixel);
        x = solver.solve(b);
    }

    void seamlessImageCloning(cv::InputArray background_, cv::InputArray foreground_, cv::InputArray foregroundMask_, int offsetX, int offsetY, cv::OutputArray destination_)
    {
        // Initialization

        cv::Mat bg = background_.getMat();
        cv::Mat fg = foreground_.getMat();
        cv::Mat fgm = foregroundMask_.getMat();

        destination_.create(bg.size(), bg.type());
        cv::Mat dst = destination_.getMat();
        bg.copyTo(dst);

        std::vector<cv::Mat> fgChannels, bgChannels, dstChannels;
        cv::split(fg, fgChannels);
        cv::split(bg, bgChannels);
        cv::split(dst, dstChannels);

        cv::Rect overlapAreaBg = cv::Rect(0, 0, bg.cols, bg.rows) | cv::Rect(offsetX, offsetY, fg.cols, fg.rows);
        cv::Rect overlapAreaFg = cv::Rect(0, 0, std::min<int>(overlapAreaBg.width, fg.cols), std::min<int>(overlapAreaBg.height, fg.rows));

        // For each channel independendly solve the linear poisson equations with boundary conditions.
        cv::Mat fgGradX, fgGradY;
        cv::Mat bgGradX, bgGradY;
        cv::Mat_<cv::Vec2f> v;
        Eigen::VectorXf solution;
                
        for (int c = 0; c < bg.channels(); ++c) {
            computeGradientImage(fgChannels[c], fgGradX, fgGradY);
            computeGradientImage(bgChannels[c], bgGradX, bgGradY);

            computeGuidanceVectorField(fgm(overlapAreaFg), bgGradX(overlapAreaBg), bgGradY(overlapAreaBg), fgGradX(overlapAreaFg), fgGradY(overlapAreaFg), v);
            solvePoissonEquations(bgChannels[c](overlapAreaBg), fgChannels[c](overlapAreaFg), fgm(overlapAreaFg), v, solution);

            // Apply solution to destination area
            cv::Mat_<uchar> dstRoi = dstChannels[c](overlapAreaBg);
            
            for (int y = 0; y < dstRoi.rows; ++y) {
                for (int x = 0, id = 0; x < dstRoi.cols; ++x, ++id) {
                    dstRoi(y, x) = cv::saturate_cast<uchar>(solution(id));
                }
            }
        }

    }
}