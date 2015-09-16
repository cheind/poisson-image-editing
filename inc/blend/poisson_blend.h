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

    /* Seamless image cloning.

       Copies the masked part of foreground onto background given the translational offset. Instead of copying 
       foreground content naively, which often leading to visible seams, the algorithm uses a method devised in 

       Pérez, Patrick, Michel Gangnet, and Andrew Blake. 
       "Poisson image editing." ACM Transactions on Graphics (TOG). Vol. 22. No. 3. ACM, 2003

       This method presented by Pérez et al. minimizes the squared error terms of the gradients of the composed 
       image and vector guidance field. The vector guidance field is modeled as mixture of gradients of foreground 
       and background image.

    */
    void seamlessImageCloning(cv::InputArray background, cv::InputArray foreground, cv::InputArray foregroundMask, int offsetX, int offsetY, cv::OutputArray destination);


}
#endif