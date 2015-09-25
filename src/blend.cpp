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
#include <blend/poisson_solver.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
namespace blend {
    
    template<class T>
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, 1> > mapChannels(cv::Mat &m, int y, int x) {
        
        typedef Eigen::Matrix<T, Eigen::Dynamic, 1> V;
        return Eigen::Map<V>(m.ptr<T>(y, x), m.channels());
    }
    
    void seamlessBlend(cv::InputArray first_,
                       cv::InputArray second_,
                       cv::InputArray mask_,
                       cv::OutputArray destination_)
    {
        
        cv::Mat first = first_.getMat();
        cv::Mat second = second_.getMat();
        
        destination_.create(first.size(), first.type());
        cv::Mat dst = destination_.getMat();
        
        // Target Laplacians are zero
        cv::Mat f(first.size(), CV_MAKE_TYPE(CV_32F, first.channels()));
        f.setTo(0);
        
        cv::Mat bm(first.size(), CV_8UC1);
        bm.setTo(constants::UNKNOWN);
        cv::Mat bv(first.size(), CV_MAKE_TYPE(CV_32F, first.channels()));
        bv.setTo(0);
        
        cv::Rect boundsSecond(0, 0, second.cols, second.rows);
        
        for (int y = 0; y < first.rows; ++y) {
            const uchar *maskRow = mask_.getMat().ptr<uchar>(y);
            for (int x = 0; x < first.cols; ++x) {
                const bool isBorder = (y == 0) || (x == 0) || (y == (first.rows - 1)) || (x == (first.cols - 1));
                const bool isFirst = (maskRow[x] == 255);
                
                if (isFirst && isBorder) {
                    bm.at<uchar>(y, x) = constants::NEUMANN_BD;
                    bv.at<float>(y, x) = 0.f;
                } else if (!isFirst && boundsSecond.contains(cv::Point(x,y))) {
                    bm.at<uchar>(y, x) = constants::DIRICHLET_BD;
                    mapChannels<float>(bv, y, x) = mapChannels<uchar>(second, y, x).cast<float>() - mapChannels<uchar>(first, y, x).cast<float>();
                } else if (!isFirst) {
                    bm.at<uchar>(y, x) = constants::DIRICHLET_BD;
                    mapChannels<float>(bv, y, x).setZero();
                }
            }
        }
        
        
        // Solve Poisson equation
        cv::Mat result;
        solvePoissonEquations(f,
                              bm,
                              bv,
                              result);
        
        cv::Mat fixed;
        first.convertTo(fixed, result.depth());
        fixed += result;
        fixed.convertTo(dst, dst.depth());
        second.copyTo(dst, (255 - mask_.getMat()));
        
        
    }
    
    
}