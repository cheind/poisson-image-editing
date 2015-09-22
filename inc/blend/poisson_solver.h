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

#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <opencv2/core/core.hpp>

namespace blend {

    namespace constants {
        const unsigned char UNKNOWN = 0;
        const unsigned char DIRICHLET_BD = 1;
        const unsigned char NEUMANN_BD = 2;
    }
     
    /**        
        Solve multi-channel Poisson equations on rectangular domain.
    */
    void solvePoissonEquations(
        cv::InputArray f,
        cv::InputArray bdMask,
        cv::InputArray bdValues,
        cv::OutputArray result);

}
#endif