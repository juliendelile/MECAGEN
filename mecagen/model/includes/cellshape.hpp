/*
 * Copyright (C) 2015 by Julien Delile
 * 
 * This file is part of MECAGEN.
 * 
 * MECAGEN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation version 3 of the License, or
 * any later version.
 * 
 * MECAGEN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MECAGEN.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef _CELLSHAPEFUNCTORS_H
#define _CELLSHAPEFUNCTORS_H

#include "define.hpp"

#include <cstring>
#include <stdio.h> 
#include <assert.h>

namespace mg {

  /** This functor calculates cell surfaces (all archetypes) and shape ratio (epithelial cells).*/
  struct evaluateCellShape
  {
    const uint*     cellType;
    d3*       cellRadius;
    const uint*     cellEpiIsPolarized;
    const double*   randomUniform;
    uint*           randomUniform_Counter;
    d3*             cellAxisAB;
    double*         cellSurface;
    double*         cellShapeRatio;           
    uint*           errorCode;

    evaluateCellShape(
                uint*     _cellType,
                d3*       _cellRadius,
                uint*     _cellEpiIsPolarized,
                double*   _randomUniform,
                uint*     _randomUniform_Counter,
                d3*       _cellAxisAB,
                double*   _cellSurface,
                double*   _cellShapeRatio,
                uint*     _errorCode
            )
         :
            cellType(_cellType),
            cellRadius(_cellRadius),
            cellEpiIsPolarized(_cellEpiIsPolarized),
            randomUniform(_randomUniform),
            randomUniform_Counter(_randomUniform_Counter),
            cellAxisAB(_cellAxisAB),
            cellSurface(_cellSurface),
            cellShapeRatio(_cellShapeRatio),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){

      uint celltype = cellType[idx];

      double a = cellRadius[idx].x;
      double c;

      // Epithelial archetype possess AB axis
      if(celltype == 2){

        // Need to instanciate an axis for spatial neighborhood algorithm
        if(cellEpiIsPolarized[idx] == 0){

          d3 axis = d3(
                randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5,
                randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5,
                randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5
                );

          axis /= length(axis);

          cellAxisAB[idx] = axis;

          cellRadius[idx].y = a;
        }
        // Polarized cell already possess an AB axis
        else{
          // cellRadius[idx].y = 1.0 * a;
          cellRadius[idx].y = 1.5 * a;
        }

        c = cellRadius[idx].y;

        //verify that the polarized epithelial cell has two defined radii
        if( c <= .0 || a <= .0){
          printf("Cell %d is epithelial and its radii are not properly specified : Rlat %lf Rab %lf.\n"
                  , idx, a, c);
          errorCode[0] = 10;
          return;
        }

        cellShapeRatio[idx] = c 
                    / sqrt(c*c + a*a);

      }
      // Other archetypes (mesenchymal, idle) does not possess AB axis.
      else{
        cellAxisAB[idx] = d3(.0);
      }


      double surface;

      // Spherical shape
      if( celltype != 2
            ||
          ( celltype == 2 && a == c )
        ){
        surface = 4.0 * PI * a * a;
      }
      else{
        //oblate spheroid
        if( c < a ){
          double e = sqrt(1 - c*c/(a*a));
          surface = 2 * PI * a * a + PI * c * c * log((1+e)/(1-e)) / e;
        }
        //prolate spheroid
        else if(c > a){
          double e = sqrt(1 - a*a/(c*c));
          surface = 2 * PI * a * a + 2 * PI * a * c * asin(e) / e;
        }
        else{
          printf("Error implementation XXX.\n");
          errorCode[0] = 11;
          return;
        }
      }
 
      cellSurface[idx] = surface;
    }
  };

}

#endif