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


#include <blend/poisson_solver.h>
#include <opencv2/opencv.hpp>
#pragma warning (push)
#pragma warning (disable: 4244)
#include <Eigen/Sparse>
#include <Eigen/Dense>
#pragma warning (pop)

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
    cv::Mat buildPixelToIndexLookup(cv::InputArray mask, uchar compareValue, int &npixel)
    {
        cv::Mat_<uchar> m = makeContinuous(mask.getMat());

        cv::Mat_<int> pixelToIndex(mask.size());
        npixel = 0;
        
        int *pixelToIndexPtr = pixelToIndex.ptr<int>();
        const uchar *maskPtr = m.ptr<uchar>();

        for (int id = 0; id < (m.rows * m.cols); ++id) {
            pixelToIndexPtr[id] = (maskPtr[id] == compareValue) ? npixel++ : -1;
        }

        return pixelToIndex;
    }
    
    void solvePoissonEquations(
        cv::InputArray f_,
        cv::InputArray dirichletMask_,
        cv::InputArray dirichletValues_,
        cv::InputArray neumannMask_,
        cv::InputArray neumannValues_,
        cv::OutputArray result_)
    {
        // Input validation

        CV_Assert(
            !f_.empty() &&
            isSameSize(f_.size(), dirichletMask_.size()) &&
            isSameSize(f_.size(), dirichletValues_.size()) &&
            isSameSize(f_.size(), neumannMask_.size()) &&
            isSameSize(f_.size(), neumannValues_.size()));

        CV_Assert(
            f_.depth() == CV_32F &&
            dirichletMask_.depth() == CV_8U &&
            neumannMask_.depth() == CV_8U &&
            dirichletValues_.depth() == CV_32F &&
            neumannValues_.depth() == CV_32F &&
            f_.channels() == dirichletValues_.channels() &&
            f_.channels() == neumannValues_.channels() &&
            dirichletMask_.channels() == 1 &&
            neumannMask_.channels() == 1);

        // We assume continuous memory on input
        cv::Mat f = makeContinuous(f_.getMat());
        cv::Mat_<uchar> dm = makeContinuous(dirichletMask_.getMat());
        cv::Mat_<uchar> nm = makeContinuous(neumannMask_.getMat());
        cv::Mat dv = makeContinuous(dirichletValues_.getMat());
        cv::Mat nv = makeContinuous(neumannValues_.getMat());

        // Allocate output
        result_.create(f.size(), f.type());
        cv::Mat r = result_.getMat();
        dv.copyTo(r);

        // The number of unknowns correspond to the number of pixels on the rectangular region 
        // that don't have a Dirichlet boundary condition.
        int nUnknowns = 0;
        cv::Mat_<int> idx = buildPixelToIndexLookup(dm, 0, nUnknowns);

        if (nUnknowns == 0) {
            // No unknowns left, we're done
            return;
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
                const int pid = idx(p);                

                if (pid == -1) {
                    // Current pixel is not an unknown, skip
                    continue;
                }

                // Start coefficients of left hand side. Based on discrete Laplacian with central difference.
                float lhs[] = { -4.f, 1.f, 1.f, 1.f, 1.f };
                int qids[5] = { -1, -1, -1, -1, -1 };
                
                for (int n = 1; n < 5; ++n) {
                    const cv::Point q(x + offsets[n][0], y + offsets[n][1]);
                    if (!bounds.contains(q)) {
                        // Neighbor is not in domain
                        lhs[center] += 1.f;
                        lhs[n] = 0.f;
                    } else {
                        // Neighbor is in domain
                        const int qid = idx(q);
                        qids[n] = qid;
                        if (dm(q)) {
                            // Neighbor has Dirichlet boundary condition applied. 
                            // Can be considered a known value, so it is moved to right hand side.
                            const float *dvs = dv.ptr<float>(q.y, q.x);
                            for (int c = 0; c < channels; ++c)
                                rhs(pid, c) -= dvs[c];
                        }
                    }
                }

                // Add f to rhs.
                const float *fs = f.ptr<float>(p.y, p.x);
                for (int c = 0; c < channels; ++c)
                    rhs(pid, c) += fs[c];

                // Build triplets.
                lhsTriplets.push_back(Eigen::Triplet<float>(pid, pid, lhs[0]));
                for (int n = 1; n < 5; ++n) {
                    if (qids[n] != -1) {
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, qids[n], lhs[n]));
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
                const int pid = idx(p);

                if (pid > -1) {
                    float *rs = r.ptr<float>(p.y, p.x);
                    for (int c = 0; c < channels; ++c)
                        rs[c] = result(pid, c);
                }

            }
        }

    }

}