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

#ifndef _MITOSIS_H
#define _MITOSIS_H

#include <cstring>
#include <stdio.h>	//printf
#include <assert.h>


#include "param.hpp"

#include "custom.hpp"

namespace mg {

  /** This function calculates the length of the cell cycle according to the selected mode.*/
  inline __device__ uint getCellCycleLength(
                                                        const CellCycleParams* ccparams, 
                                                        const double motherCycleLength,
                                                        const double* randomUniform,
                                                        uint *randomUniform_Counter
    ){

    switch (ccparams->mode)
    {
      case 0: // geometric progression
        return motherCycleLength * (ccparams->param1 + ccparams->param2 * randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0]) ]);
        // break;
      case 1: // arithmetic progression
        return motherCycleLength + ccparams->param1 * randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])];
        // break;
      case 2: // constant length
        return ccparams->param1;
        // break;
      default:
         printf("unknown cell cycle params");
         return 0;
    }

  }

  /** This functor sets the daughter cells properties when a mother cell divides.*/
  struct manage_mitosis
  {
    d3*       				      cellPosition;	    
    d3*       				      cellAxisAB;	    
    d3* 		    		        cellRadius;
    uint*   					      cellState;
    uint*   					      cellTimer;
    uint*   					      cellGeneration;
    uint*   					      cellCycleLength;
    const uint 			        timer;
    const uint 			        mPhaseLength;
    uint* 					        numCells;
    const double* 	        randomGaussian;
    uint* 					        randomGaussian_Counter;
    const double*           randomUniform;
    uint* 					        randomUniform_Counter;
    const double 		        deltaTime;
    uint* 					        cellId_bits;
    uint* 					        errorCode;
    const CustomParams*     customParams;
    const CellCycleParams*  cellCycleParams;
    double                  numLigands;
    double*                 cellLigand;
    uint*                   cellType;
    uint*                   cellEpiIsPolarized;
    d3*                     cellProtrusionExtForce;
    double*                 cellMechanoSensorQ;
    const double         numProteins;
    double*              cellProtein;

    manage_mitosis(
        d3*         		_cellPosition,
        d3*         		_cellAxis,
        d3*         		_cellRadius,
        uint*       		_cellState,
        uint*       		_cellTimer,
        uint*       		_cellGeneration,
        uint*       		_cellCycleLength,
        uint 				   _timer,
        uint 				   _mPhaseLength,
        uint* 				_numCells,
        double* 			_randomGaussian,
        uint* 				_randomGaussian_Counter,
        double* 			_randomUniform,
        uint* 				_randomUniform_Counter,
        double 				_deltaTime,
        uint* 				_cellId_bits,
        uint* 				_errorCode,
        CustomParams* _customParams,
        CellCycleParams* _cellCycleParams,
        double        _numLigands,
        double*       _cellLigand,
        uint*         _cellType,
        uint*         _cellEpiIsPolarized,
        d3*           _cellProtrusionExtForce,
        double*       _cellMechanoSensorQ,
        double        _numProteins,
        double*       _cellProtein
          )
          :
            cellPosition(_cellPosition),
            cellAxisAB(_cellAxis), 
            cellRadius(_cellRadius),            
            cellState(_cellState),            
            cellTimer(_cellTimer),            
            cellGeneration(_cellGeneration),            
            cellCycleLength(_cellCycleLength),  
            timer(_timer),
            mPhaseLength(_mPhaseLength),
            numCells(_numCells),
            randomGaussian(_randomGaussian),
            randomGaussian_Counter(_randomGaussian_Counter),
            randomUniform(_randomUniform),
            randomUniform_Counter(_randomUniform_Counter),
            deltaTime(_deltaTime),
            cellId_bits(_cellId_bits),
            errorCode(_errorCode),
            customParams(_customParams),
            cellCycleParams(_cellCycleParams),
            numLigands(_numLigands),
            cellLigand(_cellLigand),
            cellType(_cellType),
            cellEpiIsPolarized(_cellEpiIsPolarized),
            cellProtrusionExtForce(_cellProtrusionExtForce),
            cellMechanoSensorQ(_cellMechanoSensorQ),
            numProteins(_numProteins),
            cellProtein(_cellProtein)
    {}

    __device__
      void operator()(const int& idx){

        uint cellstate = cellState[idx];
        uint celltimer = cellTimer[idx];
        uint cellgeneration = cellGeneration[idx];
        uint cellcyclelength = cellCycleLength[idx];
        uint celltype = cellType[idx];

        // Cell is in mphase, cannot divide
        if(cellstate == 1){
          // if end of mphase, division is allowed
          if(timer-celltimer == mPhaseLength){
            cellState[idx] = 0;
          }
        }
        else if( (timer > 0) && ((timer-celltimer)%cellcyclelength == 0) ){

          uint sisId = mgAtomicAddOne(&numCells[0]);

          cellState[idx] = 1;
          cellState[sisId] = 1;

          cellTimer[idx] = timer;
          cellTimer[sisId] = timer;

          cellGeneration[idx] = cellgeneration + 1;
          cellGeneration[sisId] = cellgeneration + 1;

          int ccl = getCellCycleLength(&(cellCycleParams[0]), cellcyclelength, randomUniform, randomUniform_Counter);
          if(ccl<=0){
            printf("Error during mitosis cell %d: negative cell cycle.\n", idx);
            errorCode[0] = 4;
          }
          cellCycleLength[idx] = ccl;
          
          int ccl2 = getCellCycleLength(&(cellCycleParams[0]), cellcyclelength, randomUniform, randomUniform_Counter);
          if(ccl2<=0){
            errorCode[0] = 4;
          }
          cellCycleLength[sisId] = ccl2;

          // Reshape the daugther cells
          double volume_mother;
          double radius_l_mother = cellRadius[idx].x;
          double radius_ab_mother = cellRadius[idx].y;
          
          uint epiPolarized = (uint)(celltype == 2 && cellEpiIsPolarized[idx] == 1);

          if(!epiPolarized){
            volume_mother = 4.0 / 3.0 * PI * radius_l_mother * radius_l_mother * radius_l_mother;
          }
          else{
            volume_mother = 4.0 / 3.0 * PI * radius_l_mother * radius_l_mother * radius_ab_mother;
          }

          // Assume that division is symmetrical
          double volume_daugther = cellCycleParams[0].volume_ratio * volume_mother;

          if(!epiPolarized){
            double radius_l = pow( 3.0 * volume_daugther / (4.0 * PI), 1.0/3.0);
            cellRadius[idx].x = radius_l;
            cellRadius[sisId].x = radius_l;
          }
          else{
            cellRadius[sisId].y = radius_ab_mother;
            double radius_l = sqrt( 3.0 * volume_mother / (4.0 * PI * radius_ab_mother));
            cellRadius[idx].x = radius_l;
            cellRadius[sisId].x = radius_l;
          }

          //Sister cell AB axis remains identical
          d3 normal;
          if(epiPolarized){
            normal = cellAxisAB[idx];
            cellAxisAB[sisId] = normal;
          }

          //Cell are positionned in a random direction in the tangential plane
          d3 pos = cellPosition[idx];

          d3 r = d3(
                      randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5,
                      randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5,
                      randomUniform[ mgAtomicAddOne(&randomUniform_Counter[0])] - .5
                    );

          d3 mitosisAxis;
          if(epiPolarized){
            d3 u1(-normal.y, normal.x, .0);
            u1 /= length(u1);
            d3 u2 = cross(normal, u1);
            mitosisAxis = r.x * u1 + r.y * u2;
          }
          else{
            mitosisAxis = r;
          }
          mitosisAxis /= length(mitosisAxis);
          
          double dist_mit_coeff = .15;
          d3 mit_vect = dist_mit_coeff * radius_l_mother * mitosisAxis;

          cellPosition[idx] = pos + mit_vect;
          cellPosition[sisId] = pos - mit_vect;

          //Cell binary IDs
          cellId_bits[sisId] = (uint)(cellId_bits[idx] | (1<<(cellgeneration + 1)));

          // ligands quantities
          for(uint i=0; i<numLigands; i++){
            cellLigand[sisId * NUMLIGmax + i] = cellLigand[idx * NUMLIGmax + i];
          }

          cellType[sisId] = celltype;
          cellEpiIsPolarized[sisId] = cellEpiIsPolarized[idx];
          cellProtrusionExtForce[sisId] = cellProtrusionExtForce[idx];
          cellMechanoSensorQ[sisId] = cellMechanoSensorQ[idx];
          
          for(uint i=0; i<numProteins; i++){
            cellProtein[NUMPROTEINmax * sisId + i] = cellProtein[NUMPROTEINmax * idx + i];
          }
          
        }
      }	
  };

}

#endif
