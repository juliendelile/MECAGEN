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


#include "cells_producer.hpp"


namespace mg{

 void CellsProducer::doProcessing(State<HOST> *s, State<DEVICE> *sd, int step, void * buffer){
    
    Debug("CellsProducer processing");
    
    uint numCells = sd->numCells[0];

    // Cast the buffer with the forestRatio data structure
    embryoState * buf = (embryoState *) buffer;

    // Write the header values
    buf->numCells     = numCells;
    buf->gridSize     = sd->gridSize[0];
    buf->gridBoxSize  = sd->gridBoxSize[0];
    buf->worldOrigin  = sd->worldOrigin[0];
    buf->TS           = step;
    // buf->customState  = sd->customState[0];

    //the actual values are written in the buffer after the header
    void * endofHeader = (char *)buffer + sizeof(embryoState);
    buf->cellPosition = (d3*)(endofHeader);

    thrust::copy(
            sd->cellPosition.begin(),
            sd->cellPosition.begin() + numCells,
            buf->cellPosition
      );

    buf->cellAxis1 = (d3*) ((char *)(buf->cellPosition) + metaParam->numCellsMax[0] * sizeof(d3));

    thrust::copy(
            sd->cellAxis1.begin(),
            sd->cellAxis1.begin() + numCells,
            buf->cellAxis1
      );

    buf->cellAxisAB = (d3*) ((char *)(buf->cellAxis1) + metaParam->numCellsMax[0] * sizeof(d3));

    thrust::copy(
            sd->cellAxisAB.begin(),
            sd->cellAxisAB.begin() + numCells,
            buf->cellAxisAB
      );

    // for(uint i=0; i<numCells; i++){
    //   buf->cellAxisAB[i] = sd->cellCandidateAxes[NUMAXESmax * i + 1];
    // }

    buf->cellRadius = (d3*) ((char *)(buf->cellAxisAB) + metaParam->numCellsMax[0] * sizeof(d3));

    thrust::copy(
            sd->cellRadius.begin(),
            sd->cellRadius.begin() + numCells,
            buf->cellRadius
      );

    buf->cellMetricNeighbId = (uint*) ((char *)(buf->cellRadius) + metaParam->numCellsMax[0] * sizeof(d3));

    thrust::copy(
            sd->cellMetricNeighbId.begin(),
            sd->cellMetricNeighbId.begin() + numCells * metaParam->numNeighbMax[0],
            buf->cellMetricNeighbId
      );
    
    buf->cellMetricNeighbNum = (uint*) ((char *)(buf->cellMetricNeighbId) + metaParam->numCellsMax[0] * metaParam->numNeighbMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellMetricNeighbNum.begin(),
            sd->cellMetricNeighbNum.begin() + numCells,
            buf->cellMetricNeighbNum
      );

    buf->cellTopologicalNeighbId = (uint*) ((char *)(buf->cellMetricNeighbNum) + metaParam->numCellsMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellTopologicalNeighbId.begin(),
            sd->cellTopologicalNeighbId.begin() + numCells * NUMNEIGHBTOPOmax,
            buf->cellTopologicalNeighbId
      );
    
    buf->cellTopologicalNeighbNum = (uint*) ((char *)(buf->cellTopologicalNeighbId) + metaParam->numCellsMax[0] * NUMNEIGHBTOPOmax * sizeof(uint));

    thrust::copy(
            sd->cellTopologicalNeighbNum.begin(),
            sd->cellTopologicalNeighbNum.begin() + numCells,
            buf->cellTopologicalNeighbNum
      );

    // std::cout << "producer : ";
    // for(uint i=0; i < sd->cellTopologicalNeighbNum[1]; i++){
    //     std::cout << " " << sd->cellTopologicalNeighbId[1 * metaParam->numNeighbMax[0] + i];
    // }
    // std::cout << std::endl;

    buf->cellPopulation = (uint*) ((char *)(buf->cellTopologicalNeighbNum) + metaParam->numCellsMax[0] * sizeof(uint));

    #if WADD
        thrust::copy(
                sd->cellWaddingtonianType.begin(),
                sd->cellWaddingtonianType.begin() + numCells,
                buf->cellPopulation
          );
    #endif

    buf->cellId_bits = (uint*) ((char *)(buf->cellPopulation) + metaParam->numCellsMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellId_bits.begin(),
            sd->cellId_bits.begin() + numCells,
            buf->cellId_bits
      );

    buf->cellTimer = (uint*) ((char *)(buf->cellId_bits) + metaParam->numCellsMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellTimer.begin(),
            sd->cellTimer.begin() + numCells,
            buf->cellTimer
      );

    buf->cellGeneration = (uint*) ((char *)(buf->cellTimer) + metaParam->numCellsMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellGeneration.begin(),
            sd->cellGeneration.begin() + numCells,
            buf->cellGeneration
      );

    buf->cellLigand = (double*) ((char *)(buf->cellGeneration) + metaParam->numCellsMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellLigand.begin(),
            sd->cellLigand.begin() + numCells * NUMLIGmax,
            buf->cellLigand
      );

    buf->cellType = (uint*) ((char *)(buf->cellLigand) + metaParam->numCellsMax[0] * NUMLIGmax * sizeof(uint));

    thrust::copy(
            sd->cellType.begin(),
            sd->cellType.begin() + numCells,
            buf->cellType
    );

    buf->cellEpiIsPolarized = (uint*) ((char *)(buf->cellType) + metaParam->numCellsMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellEpiIsPolarized.begin(),
            sd->cellEpiIsPolarized.begin() + numCells,
            buf->cellEpiIsPolarized
    );

    buf->cellEpiId = (uint*) ((char *)(buf->cellEpiIsPolarized) + metaParam->numCellsMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellEpiId.begin(),
            sd->cellEpiId.begin() + numCells,
            buf->cellEpiId
    );

    buf->cellCandidateAxes = (d3*) ((char *)(buf->cellEpiId) + metaParam->numCellsMax[0] * sizeof(uint));

    thrust::copy(
            sd->cellCandidateAxes.begin(),
            sd->cellCandidateAxes.begin() + NUMAXESmax * numCells,
            buf->cellCandidateAxes
    );
    
    buf->cellProtein = (double*) ((char *)(buf->cellCandidateAxes) + metaParam->numCellsMax[0] * NUMAXESmax * sizeof(d3));

    thrust::copy(
            sd->cellProtein.begin(),
            sd->cellProtein.begin() + numCells * NUMPROTEINmax,
            buf->cellProtein
    );

   

    buf->customStateBuffer.copy(&(sd->customState), numCells);

    

  }

  size_t CellsProducer::getRequestedBufferSize(){
    // Here the size of the buffer is equal to the size of the header (i.e. the size of a int pointer) plus
    // the size of the whole forest_value integer array
    return      sizeof(embryoState) +
                metaParam->numCellsMax[0] * sizeof(d3) +
                metaParam->numCellsMax[0] * sizeof(d3) +
                metaParam->numCellsMax[0] * sizeof(d3) +
                metaParam->numCellsMax[0] * sizeof(d3) +
                metaParam->numCellsMax[0] * metaParam->numNeighbMax[0] * sizeof(uint) +
                metaParam->numCellsMax[0] * sizeof(uint) +
                metaParam->numCellsMax[0] * NUMNEIGHBTOPOmax * sizeof(uint) +
                metaParam->numCellsMax[0] * sizeof(uint) + 
                metaParam->numCellsMax[0] * sizeof(uint) + 
                metaParam->numCellsMax[0] * sizeof(uint) + 
                metaParam->numCellsMax[0] * sizeof(uint) + 
                metaParam->numCellsMax[0] * sizeof(uint) +
                metaParam->numCellsMax[0] * NUMLIGmax * sizeof(double) +
                metaParam->numCellsMax[0] * sizeof(uint) +
                metaParam->numCellsMax[0] * sizeof(uint) +
                metaParam->numCellsMax[0] * sizeof(uint) +
                metaParam->numCellsMax[0] * NUMAXESmax * sizeof(d3) +
                metaParam->numCellsMax[0] * NUMPROTEINmax * sizeof(double)
                ;
  }

}