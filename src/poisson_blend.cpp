/**
 This file is part of PoissonBlend.
 
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


#include <blend/poisson_blend.h>
#include <opencv2/opencv.hpp>
#pragma warning (push)
#pragma warning (disable: 4244)
#include <Eigen/Sparse>
#include <Eigen/Dense>
#pragma warning (pop)

namespace blend {
    
    namespace detail {
        bool findOverlap(cv::InputArray background,
                         cv::InputArray foreground,
                         int offsetX, int offsetY,
                         cv::Rect &rBackground,
                         cv::Rect &rForeground)
        {
            cv::Mat bg = background.getMat();
            cv::Mat fg = foreground.getMat();
            
            rBackground = cv::Rect(0, 0, bg.cols, bg.rows) & cv::Rect(offsetX, offsetY, fg.cols, fg.rows);
            rForeground = cv::Rect(0, 0, std::min<int>(rBackground.width, fg.cols), std::min<int>(rBackground.height, fg.rows));
            
            return rForeground.area() > 0;
            
        }
        
        void computeMixedGradientVectorField(cv::InputArray background,
                                             cv::InputArray foreground,
                                             cv::InputArray foregroundMask,
                                             cv::OutputArray vx_,
                                             cv::OutputArray vy_)
        {
            cv::Mat bg = background.getMat();
            cv::Mat fg = foreground.getMat();
            
            const int channels = bg.channels();
            
            vx_.create(bg.size(), CV_MAKETYPE(CV_32F, channels));
            vy_.create(bg.size(), CV_MAKETYPE(CV_32F, channels));
            
            cv::Mat vx = vx_.getMat();
            cv::Mat vy = vy_.getMat();
            
            cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
            cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
            
            cv::Mat vxf, vyf, vxb, vyb;
            cv::filter2D(fg, vxf, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
            cv::filter2D(fg, vyf, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
            cv::filter2D(bg, vxb, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
            cv::filter2D(bg, vyb, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
            
            
            for(int id = 0; id <= (vx.rows * vx.cols * channels - channels); ++id)
            {
                const cv::Vec2f g[2] = {
                    cv::Vec2f(vxf.ptr<float>()[id], vyf.ptr<float>()[id]),
                    cv::Vec2f(vxb.ptr<float>()[id], vyb.ptr<float>()[id])
                };
                
                int which = (g[0].dot(g[0]) > g[1].dot(g[1])) ? 0 : 1;
                
                vx.ptr<float>()[id] = g[which][0];
                vy.ptr<float>()[id] = g[which][1];
            }
        }
        
        void computeWeightedGradientVectorField(cv::InputArray background,
                                                cv::InputArray foreground,
                                                cv::InputArray foregroundMask,
                                                cv::OutputArray vx,
                                                cv::OutputArray vy,
                                                float weightForeground)
        {
            
            cv::Mat bg = background.getMat();
            cv::Mat fg = foreground.getMat();
            
            cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
            cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
            
            cv::Mat vxf, vyf, vxb, vyb;
            cv::filter2D(fg, vxf, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
            cv::filter2D(fg, vyf, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
            cv::filter2D(bg, vxb, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
            cv::filter2D(bg, vyb, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
            
            cv::addWeighted(vxf, weightForeground, vxb, 1.f - weightForeground, 0, vx);
            cv::addWeighted(vyf, weightForeground, vyb, 1.f - weightForeground, 0, vy);
        }
        
        void solvePoissonEquationsSingleChannel(cv::Mat_<uchar> bg,
                                                cv::Mat_<uchar> fg,
                                                cv::Mat_<uchar> fgm,
                                                cv::Mat_<float> vx,
                                                cv::Mat_<float> vy,
                                                cv::Mat dst)
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
            
            const int neighbors[4][2] = { {0, -1}, {0, 1}, {-1, 0}, {1, 0 } };
            
            
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
        
        
        void solvePoissonEquations(cv::InputArray background,
                                   cv::InputArray foreground,
                                   cv::InputArray foregroundMask,
                                   cv::InputArray vx,
                                   cv::InputArray vy,
                                   cv::OutputArray destination)
        {
            // Split the input into separate channels, then invoke the solving
            // for each channel and merge computed channel values into destination.
            
            cv::Mat bg = background.getMat();
            cv::Mat fg = foreground.getMat();
            cv::Mat fgm = foregroundMask.getMat();
            
            destination.create(bg.size(), bg.type());
            cv::Mat dst = destination.getMat();
            
            bg.copyTo(dst);
            
            std::vector<cv::Mat> bgChannels, fgChannels, dstChannels, vxChannels, vyChannels;
            
            cv::split(fg, fgChannels);
            cv::split(bg, bgChannels);
            cv::split(vx, vxChannels);
            cv::split(vy, vyChannels);
            cv::split(dst, dstChannels);
            
            for (int c = 0; c < bg.channels(); ++c) {
                
                solvePoissonEquationsSingleChannel(bgChannels[c],
                                                   fgChannels[c],
                                                   fgm,
                                                   vxChannels[c], vyChannels[c],
                                                   dstChannels[c]);
            }
            
            cv::merge(dstChannels, dst);
            
        }
    }
    
    void seamlessClone(cv::InputArray background,
                       cv::InputArray foreground,
                       cv::InputArray foregroundMask,
                       int offsetX,
                       int offsetY,
                       cv::OutputArray destination,
                       CloneType type)
    {
        
        // We want to ensure that the mask does not have white pixels at image edges. We
        // need at least a one pixel black border for the Dirichlet boundary conditions.
        cv::Mat modifiedMask = foregroundMask.getMat().clone();
        cv::threshold(modifiedMask, modifiedMask, 1, 255, cv::THRESH_BINARY);
        cv::rectangle(modifiedMask, cv::Rect(0, 0, modifiedMask.cols - 1, modifiedMask.rows - 1), 0, 1);
        
        // Copy original background as we only solve for the overlapping area of the translated foreground mask.
        background.getMat().copyTo(destination);
        
        // Find overlapping region. We will only perform on this region
        cv::Rect rbg, rfg;
        if (!detail::findOverlap(background, foreground, offsetX, offsetY, rbg, rfg))
            return;

        // Compute the guidance vector field
        cv::Mat vx, vy;
        switch (type) {
            case CLONE_FOREGROUND_GRADIENTS:
                detail::computeWeightedGradientVectorField(background.getMat()(rbg),
                                                           foreground.getMat()(rfg),
                                                           modifiedMask(rfg),
                                                           vx, vy,
                                                           1.f);
                break;
                
            case CLONE_AVERAGED_GRADIENTS:
                detail::computeWeightedGradientVectorField(background.getMat()(rbg),
                                                           foreground.getMat()(rfg),
                                                           modifiedMask(rfg),
                                                           vx, vy,
                                                           0.5f);
                break;
                
            case CLONE_MIXED_GRADIENTS:
                detail::computeMixedGradientVectorField(background.getMat()(rbg),
                                                        foreground.getMat()(rfg),
                                                        modifiedMask(rfg),
                                                        vx, vy);
                break;
                
            default:
                break;
        }
        
        // Solve equations for each channel separately.
        detail::solvePoissonEquations(background.getMat()(rbg),
                                      foreground.getMat()(rfg),
                                      modifiedMask(rfg),
                                      vx, vy,
                                      destination.getMat()(rbg));
    }
    
    
}