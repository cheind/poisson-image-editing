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


#include <blend/poisson_solver.h>
#include <opencv2/opencv.hpp>
#pragma warning (push)
#pragma warning (disable: 4244)
#include <Eigen/Sparse>
#include <Eigen/Dense>
#pragma warning (pop)
#include <bitset>

namespace blend {       

    bool isSameSize(cv::Size a, cv::Size b) {
        return a.width == b.width && a.height == b.height;
    }

    /* Make matrix memory continuous. */
    cv::Mat makeContinuous(cv::Mat m) {       
        if (!m.isContinuous()) {
            m = m.clone();
        }        
        return m;
    }

    /* Build a one dimensional index lookup for element in mask. */
    cv::Mat buildPixelToIndexLookup(cv::InputArray mask, int &npixel)
    {
        cv::Mat_<uchar> m = makeContinuous(mask.getMat());

        cv::Mat_<int> pixelToIndex(mask.size());
        npixel = 0;
        
        int *pixelToIndexPtr = pixelToIndex.ptr<int>();
        const uchar *maskPtr = m.ptr<uchar>();

        for (int id = 0; id < (m.rows * m.cols); ++id) {
            pixelToIndexPtr[id] = (maskPtr[id] == constants::DIRICHLET_BD) ? -1 : npixel++;
        }

        return pixelToIndex;
    }
    
    void solvePoissonEquations(
        cv::InputArray f_,
        cv::InputArray bdMask_,
        cv::InputArray bdValues_,
        cv::OutputArray result_)
    {
        // Input validation

        CV_Assert(
            !f_.empty() &&
            isSameSize(f_.size(), bdMask_.size()) &&
            isSameSize(f_.size(), bdValues_.size())
        );

        CV_Assert(
            f_.depth() == CV_32F &&
            bdMask_.depth() == CV_8U &&
            bdValues_.depth() == CV_32F &&
            f_.channels() == bdValues_.channels() &&
            bdMask_.channels() == 1);

        // We assume continuous memory on input
        cv::Mat f = makeContinuous(f_.getMat());
        cv::Mat_<uchar> bm = makeContinuous(bdMask_.getMat());
        cv::Mat bv = makeContinuous(bdValues_.getMat());

        // Allocate output
        result_.create(f.size(), f.type());
        cv::Mat r = result_.getMat();
        bv.copyTo(r, bm == constants::DIRICHLET_BD);

        // The number of unknowns correspond to the number of pixels on the rectangular region 
        // that don't have a Dirichlet boundary condition.
        int nUnknowns = 0;
        cv::Mat_<int> unknownIdx = buildPixelToIndexLookup(bm, nUnknowns);

        if (nUnknowns == 0) {
            // No unknowns left, we're done
            return;
        } else if (nUnknowns == f.size().area()) {
            // All unknowns, will not lead to a unique solution
            // TODO emit warning
        }

        const cv::Rect bounds(0, 0, f.cols, f.rows);

        // Directional indices
        const int center = 0;
        const int north = 1;
        const int east = 2;
        const int south = 3;
        const int west = 4;

        // Neighbor offsets in all directions
        const int offsets[5][2] = { { 0, 0 }, { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };
        
        // Directional opposite
        const int opposite[5] = { center, south, west, north, east };
        const int channels = f.channels();
        
        std::vector< Eigen::Triplet<float> > lhsTriplets;
        lhsTriplets.reserve(nUnknowns * 5);

        Eigen::MatrixXf rhs(nUnknowns, channels);
        rhs.setZero();
        
        // Loop over domain once. The coefficient matrix A is the same for all
        // channels, the right hand side is channel dependent.

        for (int y = 0; y < f.rows; ++y) {
            for (int x = 0; x < r.cols; ++x) {

                const cv::Point p(x, y);
                const int pid = unknownIdx(p);

                if (pid == -1) {
                    // Current pixel is not an unknown, skip
                    continue;
                }

                // Start coefficients of left hand side. Based on discrete Laplacian with central difference.
                float lhs[] = { -4.f, 1.f, 1.f, 1.f, 1.f };
                
                const bool hasNeumann = (bm(p) == constants::NEUMANN_BD);
                
                if (hasNeumann) {
                    
                    // Implementation note:
                    //
                    // We first sweep over all neighbors and apply Neumann boundary (NB) conditions if necessary.
                    // NBs are currently only applied if the neighbor is not in the domain or it has Dirichlet
                    // boundary condition (DB).
                    //
                    // When the neighbor is not available we introduce ghost points which are immediately
                    // removed by substitution. Assume that we are at a pixel C at the top border (not corner)
                    // and that pixel is assigned a NB = 1. Denoting the pixels C, N, E, S, W we have for C
                    // the Laplacian
                    //      1: -4C + N + E + S + W = f(x)
                    // From NB we have
                    //      2: (N - S) * 0.5 = 1
                    // As N is not in the domain we need to get rid of it through substitution. Rewriting 2:
                    //      N = 2 + S
                    // and substituting in 1:
                    //      -4C + (2 + S) + E + S + W = f(x)
                    //      -4C + E + 2S + W = f(x) - 2
                    
                    for (int n = 1; n < 5; ++n) {
                        const cv::Point q(x + offsets[n][0], y + offsets[n][1]);
                        
                        if (!bounds.contains(q) || bm(q) == constants::DIRICHLET_BD) {
                            lhs[opposite[n]] += 1.0f;
                            lhs[n] = 0.f;
                            rhs.row(pid) += 2.f * Eigen::Map<Eigen::VectorXf>(bv.ptr<float>(p.y, p.x), channels);
                        }
                    }
                }
                
                for (int n = 1; n < 5; ++n) {
                    const cv::Point q(x + offsets[n][0], y + offsets[n][1]);
                    
                    const bool hasNeighbor = bounds.contains(q);
                    const bool isNeighborDirichlet = hasNeighbor && (bm(q) == constants::DIRICHLET_BD);
                    
                    if (!hasNeumann && !hasNeighbor) {
                        lhs[center] += lhs[n];
                        lhs[n] = 0.f;
                    } else if (isNeighborDirichlet) {
                        
                        // Implementation note:
                        //
                        // Dirichlet boundary conditions (DB) turn neighbor unknowns into knowns (data) and
                        // are therefore moved to the right hand side.
                        
                        rhs.row(pid) -= lhs[n] * Eigen::Map<Eigen::VectorXf>(bv.ptr<float>(q.y, q.x), channels);
                        lhs[n] = 0.f;
                    }
                }


                // Add f to rhs.
                rhs.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(p.y, p.x), channels);

                // Build triplets for row              
                for (int n = 0; n < 5; ++n) {
                    if (lhs[n] != 0.f) {
                        const cv::Point q(x + offsets[n][0], y + offsets[n][1]);
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, unknownIdx(q), lhs[n]));
                    }
                }
                    
            }
        }

        // Solve the sparse linear system of equations

        Eigen::SparseMatrix<float> A(nUnknowns, nUnknowns);
        A.setFromTriplets(lhsTriplets.begin(), lhsTriplets.end());

        Eigen::SparseLU< Eigen::SparseMatrix<float> > solver;
        solver.analyzePattern(A);
        solver.factorize(A);

        Eigen::MatrixXf result(nUnknowns, channels);
        for (int c = 0; c < channels; ++c)
            result.col(c) = solver.solve(rhs.col(c));
        

        // Copy results back

        for (int y = 0; y < f.rows; ++y) {
            for (int x = 0; x < f.cols; ++x) {
                const cv::Point p(x, y);
                const int pid = unknownIdx(p);

                if (pid > -1) {
                    Eigen::Map<Eigen::VectorXf>(r.ptr<float>(p.y, p.x), channels) = result.row(pid);
                }

            }
        }

    }

}