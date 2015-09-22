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
        cv::Mat_<int> unknownIdx = buildPixelToIndexLookup(dm, 0, nUnknowns);

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
        
        std::bitset<3> state;

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
                int qids[5] = { pid, -1, -1, -1, -1 };

                state.set(2, (nm(p) > 0));
                
                for (int n = 1; n < 5; ++n) {
                    const cv::Point q(x + offsets[n][0], y + offsets[n][1]);
                    const cv::Point qo(x + offsets[opposite[n]][0], y + offsets[opposite[n]][1]);

                    state.set(1, bounds.contains(q));
                    state.set(0, bounds.contains(qo));

                    switch (state.to_ulong())
                    {
                    case 0: //000
                    case 1: //001
                    case 4: //100
                        // Decrease the number of unknowns
                        // Neighbor is not in domain and we don't have a Neumann boundary, 
                        // or we have Neumann boundary but the neighbor and opposite neighbor are
                        // both not in domain.
                        lhs[center] += 1.f;
                        lhs[n] = 0.f;
                        break;

                    case 2: //010
                    case 3: //011
                    case 6: //110
                    case 7: //111
                        // Apply Dirichlet boundary condition or keep unknown
                        // We don't have a Neumann boundary and neighbor is in domain.
                        qids[n] = unknownIdx(q);
                        lhs[n] += (qids[n] == -1) ? -1.f : 0.f;
                        if (lhs[n] == 0) {
                            // Neighbor has Dirichlet boundary condition applied. 
                            // Can be considered a known value, so it is moved to right hand side.
                            rhs.row(pid) -= Eigen::Map<Eigen::VectorXf>(dv.ptr<float>(q.y, q.x), channels);
                        }

                        break;

                    case 5: //101
                        // Apply Neumann boundary condition on rectangular domain boundaries.                        
                        lhs[n] -= 1.f;
                        lhs[opposite[n]] += 1.f;
                        rhs.row(pid) -= Eigen::Map<Eigen::VectorXf>(nv.ptr<float>(p.y, p.x), channels);

                        break;
                    }
                }

                // Add f to rhs.
                rhs.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(p.y, p.x), channels);

                // Build triplets for row              
                for (int n = 0; n < 5; ++n) {
                    if (lhs[n] != 0) {
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
                const int pid = unknownIdx(p);

                if (pid > -1) {
                    Eigen::Map<Eigen::VectorXf>(r.ptr<float>(p.y, p.x), channels) = result.row(pid);
                }

            }
        }

    }

}