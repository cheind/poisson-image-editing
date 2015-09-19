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

#ifndef POISSON_BLEND_H
#define POISSON_BLEND_H

#include <opencv2/core/core.hpp>

namespace blend {
    
    namespace detail {
    
        /**
         
         Determine the area of overlap.
         
         Computes the areas of overlap for background and foreground when foreground
         is layed over background given a translational offset.
         
         */
        bool findOverlap(cv::InputArray background,
                         cv::InputArray foreground,
                         int offsetX, int offsetY,
                         cv::Rect &rBackground,
                         cv::Rect &rForeground);
        
        /** 
         
         Compute Poisson guidance vector field by mixing gradients from background and foreground.
         
         */
        void computeMixedGradientVectorField(cv::InputArray background,
                                             cv::InputArray foreground,
                                             cv::InputArray foregroundMask,
                                             cv::OutputArray vx,
                                             cv::OutputArray vy);
        
        /**
         
         Compute Poisson guidance vector field by averaging background and foreground gradients.
         
         */
        void computeWeightedGradientVectorField(cv::InputArray background,
                                                cv::InputArray foreground,
                                                cv::InputArray foregroundMask,
                                                cv::OutputArray vx,
                                                cv::OutputArray vy,
                                                float weightForeground);
        
        /** 
         
         Solve multi-channel Poisson equations.
         
         */
        void solvePoissonEquations(cv::InputArray background,
                                   cv::InputArray foreground,
                                   cv::InputArray foregroundMask,
                                   cv::InputArray vx,
                                   cv::InputArray vy,
                                   cv::OutputArray destination);
    }
    
    /** 
    
     Defines the available cloning variants.
     
     */
    enum CloneType {
        CLONE_FOREGROUND_GRADIENTS,
        CLONE_AVERAGED_GRADIENTS,
        CLONE_MIXED_GRADIENTS
    };
    
    /**
     
     Seamless image cloning.
     
     Copies the masked part of foreground onto background given a translational offset. Instead of copying
     foreground content naively, which often leading to visible seams, the algorithm uses a method devised in
     
     Pèrez, Patrick, Michel Gangnet, and Andrew Blake.
     "Poisson image editing." ACM Transactions on Graphics (TOG). Vol. 22. No. 3. ACM, 2003
     
     This method minimizes the squared error terms of image gradients and a vector guidance field. The optimization 
     is subject to Dirichlet boundary conditions which fix intensities at mask borders.
     
     If the vector guidance field is the gradient of the foreground image, then the solution is given by
        Laplacian(result) = divergence(gradient(foreground)) s.t. boundary conditions.
     
     Depending on the vector guidance field differents blend effects can be generated.
     
     */
    void seamlessClone(cv::InputArray background,
                       cv::InputArray foreground,
                       cv::InputArray foregroundMask,
                       int offsetX,
                       int offsetY,
                       cv::OutputArray destination,
                       CloneType type);

}
#endif