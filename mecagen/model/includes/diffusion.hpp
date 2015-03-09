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

#ifndef _DIFFUSION_H
#define _DIFFUSION_H

namespace mg {

  /** This functor calculates which ligand quantity is transfered between neighbor cells. */
  struct diffuseLigands
  {
    const uint            numLigands;        
    const uint*           cellTopologicalNeighbNum;  
    const uint*           cellTopologicalNeighbId;
    const LigandParams*   ligandParams;
    const double*         cellLigand;
    double*               cellLigandUpdate;
    const d3*             cellPosition;
    const double*         cellContactSurfaceArea;
    const double          deltaTime;
    
    diffuseLigands(
                uint           _numLigands,        
                uint*          _cellTopologicalNeighbNum,  
                uint*          _cellTopologicalNeighbId,
                LigandParams*  _ligandParams,
                double*        _cellLigand,
                double*        _cellLigandUpdate,
                d3*            _cellPosition,
                double*        _cellContactSurfaceArea,
                double         _deltaTime
            )
         :
            numLigands(_numLigands),
            cellTopologicalNeighbNum(_cellTopologicalNeighbNum),
            cellTopologicalNeighbId(_cellTopologicalNeighbId),
            ligandParams(_ligandParams),
            cellLigand(_cellLigand),
            cellLigandUpdate(_cellLigandUpdate),
            cellPosition(_cellPosition), 
            cellContactSurfaceArea(_cellContactSurfaceArea),
            deltaTime(_deltaTime)
            {}

    __device__
    void operator()(const int& idx){
      
      uint numNeighb  = cellTopologicalNeighbNum[idx];
      d3 pos1         = cellPosition[idx];
        
      //store cell idx ligand in register memory (CUDA version)
      double cellLigQ[2*NUMLIGmax];
      for(uint i=0;i<numLigands;i++){
        cellLigQ[2*i]   = cellLigand[NUMLIGmax*idx + i];
        cellLigQ[2*i+1] = .0;
      }

      for(uint i=0; i<numNeighb; i++){
        
        uint topoNeighbIndex    = idx * NUMNEIGHBTOPOmax + i;
        uint neighbCellId   = cellTopologicalNeighbId[topoNeighbIndex];

        d3 pos2     = cellPosition[neighbCellId];
        d3 relPos   = pos2 - pos1;
        double dist     = length(relPos);
       
        double surface  = cellContactSurfaceArea[topoNeighbIndex];
        
        uint numNeighbOfNeighb = cellTopologicalNeighbNum[neighbCellId];

        double maxNeighb = (numNeighb>numNeighbOfNeighb)?(double)numNeighb:(double)numNeighbOfNeighb;
        
        double update;

        for(uint j=0; j<numLigands; j++){
          
          double diffcoeff = ligandParams[j].diffusion; 
          
          double neighbLigQ = cellLigand[ NUMLIGmax * neighbCellId + j];

          if( cellLigQ[2*j] > neighbLigQ ){
          
            update = .5 *  1.0/maxNeighb * diffcoeff * (cellLigQ[2*j] - neighbLigQ) * surface / dist;
          
            if(update > .5 *  1.0/maxNeighb * cellLigQ[2*j]){
              update = .5 *  1.0/maxNeighb * cellLigQ[2*j];
            }
            
            update = -update;
          }
          else{
            
            update = .5 *  1.0/maxNeighb * diffcoeff * (neighbLigQ - cellLigQ[2*j]) * surface / dist;
            
            if(update > .5 * 1.0/maxNeighb * neighbLigQ){
              update = .5 * 1.0/maxNeighb * neighbLigQ;
            }
            
          }
          
          cellLigQ[2*j+1] += update * deltaTime;
        }
      }

      for(uint i=0;i<numLigands;i++){
        // Degradation
        cellLigQ[2*i+1] -= ligandParams[i].kappa * cellLigQ[2*i] * deltaTime;

        cellLigandUpdate[NUMLIGmax*idx + i] = cellLigQ[2*i+1];
      }
    }
  };

  /**This functor integrate ligand concentrations after update by diffusion, secretion and transduction.*/
  struct updateLigands
  {
    const uint            numLigands;        
    const LigandParams*   ligandParams;
    double*               cellLigand;
    const double*         cellLigandUpdate;
    
    updateLigands(
                uint           _numLigands,        
                LigandParams*  _ligandParams,
                double*        _cellLigand,
                double*        _cellLigandUpdate
            )
         :
            numLigands(_numLigands),
            ligandParams(_ligandParams),
            cellLigand(_cellLigand),
            cellLigandUpdate(_cellLigandUpdate)
            {}

    __device__
    void operator()(const int& idx){
      
      for(uint i=0;i<numLigands;i++){
        cellLigand[NUMLIGmax*idx + i] = cellLigand[NUMLIGmax*idx + i] + cellLigandUpdate[NUMLIGmax*idx + i];
      }
      
    }
  };
  
} //end namespace

#endif
