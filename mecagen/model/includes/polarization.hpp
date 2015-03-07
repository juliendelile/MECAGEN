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

#ifndef _POLARIZATION_H
#define _POLARIZATION_H

namespace mg {

  /** This functor evaluates the candidate polarization axes' update values.*/
  struct computePolarizationAxes
  {
    const uint                      numPolarizationAxes;        
    const uint                      numLigands; 
    const PolarizationAxisParams*   polarizationAxisParams;       
    const uint*                     cellTopologicalNeighbNum;  
    const uint*                     cellTopologicalNeighbId;
    const double*                   cellLigand;
    const d3*                       cellPosition;
    const d3*                       cellCandidateAxes;
    const uint*                     cellType;
    const d3*                       cellAxisAB;
    const uint*                     cellEpiIsPolarized;
    d3*                             cellCandidateAxesUpdate;
    
    computePolarizationAxes(
                uint                    _numPolarizationAxes,        
                uint                    _numLigands,  
                PolarizationAxisParams* _polarizationAxisParams,      
                uint*                   _cellTopologicalNeighbNum,  
                uint*                   _cellTopologicalNeighbId,
                double*                 _cellLigand,
                d3*                     _cellPosition,
                d3*                     _cellCandidateAxes,
                uint*                   _cellType,
                d3*                     _cellAxisAB,
                uint*                   _cellEpiIsPolarized,
                d3*                     _cellCandidateAxesUpdate
            )
         :
            numPolarizationAxes(_numPolarizationAxes),
            numLigands(_numLigands),
            polarizationAxisParams(_polarizationAxisParams),
            cellTopologicalNeighbNum(_cellTopologicalNeighbNum),
            cellTopologicalNeighbId(_cellTopologicalNeighbId),
            cellLigand(_cellLigand),
            cellPosition(_cellPosition), 
            cellCandidateAxes(_cellCandidateAxes),
            cellType(_cellType),
            cellAxisAB(_cellAxisAB),
            cellEpiIsPolarized(_cellEpiIsPolarized),
            cellCandidateAxesUpdate(_cellCandidateAxesUpdate)
            {}

    __device__
    void operator()(const int& idx){
      
      uint numNeighb  = cellTopologicalNeighbNum[idx];
      d3 pos1         = cellPosition[idx];

      d3 axes[NUMAXESmax];
      for(uint i=0; i<numPolarizationAxes; i++){
        axes[i] = d3(.0);
      }

      double cellLigQ[NUMLIGmax];
      for(uint i=0;i<numLigands;i++){
        cellLigQ[i] = cellLigand[NUMLIGmax*idx + i];
      }

      double diffqligand;

      PolarizationAxisParams pap[NUMAXESmax];
      for(uint i=0; i<numPolarizationAxes; i++){
        pap[i] = polarizationAxisParams[i];
      }

      double axisMode2max[NUMAXESmax];
      for(uint i=0; i<numPolarizationAxes; i++){
        if(pap[i].compMode == 2){
          axisMode2max[i] = cellLigQ[ pap[i].idlig ];
        }
      }

      for(uint i=0; i<numNeighb; i++){

        uint topoNeighbIndex    = idx * NUMNEIGHBTOPOmax + i;
        uint neighbCellId   = cellTopologicalNeighbId[topoNeighbIndex];

        d3 relPos   = cellPosition[neighbCellId] - pos1;
        relPos /= length(relPos);

        for(uint j=0; j < numPolarizationAxes; j++){

          double neighbCellLigand = cellLigand[NUMLIGmax*neighbCellId + pap[j].idlig];

          if(pap[j].compMode == 0){      //weighted neighboring link
            diffqligand = neighbCellLigand - cellLigQ[pap[j].idlig];
            axes[j] += diffqligand * diffqligand * diffqligand * relPos;
          }
          else if(pap[j].compMode == 1){ //weighted average
            if( neighbCellLigand > pap[j].param1 ){
              axes[j] += neighbCellLigand * cellCandidateAxes[NUMAXESmax * neighbCellId + j];
            }
          }
          else if(pap[j].compMode == 2){ //maximum
            if( neighbCellLigand > axisMode2max[j] ){
              
              axisMode2max[j] = neighbCellLigand;
              axes[j] = relPos;
            }
          }

        }
      }

      /****************************************
      /* Mode weighted average initialization *
      /***************************************/

      for(uint j=0; j<numPolarizationAxes; j++){
        if(pap[j].compMode == 1 && length(axes[j]) == .0){
          for(uint i=0; i<numNeighb; i++){

            uint topoNeighbIndex    = idx * NUMNEIGHBTOPOmax + i;
            uint neighbCellId       = cellTopologicalNeighbId[topoNeighbIndex];
            d3 relPos               = cellPosition[neighbCellId] - pos1;
            relPos /= length(relPos);

            double neighbCellLigand = cellLigand[NUMLIGmax*neighbCellId + pap[j].idlig];
            
            if( neighbCellLigand > pap[j].param1 ){
              axes[j] += relPos * ( neighbCellLigand - cellLigQ[pap[j].idlig]);
            }
          }
        }
      }

      /****************************************
      /****** Projection in Epithelial cells **
      /***************************************/

      uint epiPolarized = (uint)(cellType[idx] == 2 && cellEpiIsPolarized[idx] == 1);
      
      d3 ABaxis;
      if(epiPolarized){
        ABaxis = cellAxisAB[idx];
      }

      for(uint i=0; i< numPolarizationAxes; i++){

        // If the cell is epithelial, its internal polarization is either oriented along the AB axis, or orthogonal to it. 
        if(epiPolarized){
          // apico-basal polarization case
          // no reorientation
          if(pap[i].apicoBasalInEpithelium){
            // axes[i] = ABaxis;
          }
          // lateral polarization case, projection on the orthogonal plane
          else{
            double scal = dot(axes[i], ABaxis);
            axes[i] -= scal * ABaxis;
          }
        }
        
        cellCandidateAxesUpdate[NUMAXESmax * idx + i] = axes[i];
      }

    } // end operator()

  }; // end functor

  /** This functor integrates the candidate polarization axes.*/
  struct updatePolarizationAxes
  {
    const uint                      numPolarizationAxes;     
    d3*                       cellCandidateAxes;
    const d3*                             cellCandidateAxesUpdate;
    
    updatePolarizationAxes(
                uint                    _numPolarizationAxes,
                d3*                     _cellCandidateAxes,
                d3*                     _cellCandidateAxesUpdate
            )
         :
            numPolarizationAxes(_numPolarizationAxes),
            cellCandidateAxes(_cellCandidateAxes),
            cellCandidateAxesUpdate(_cellCandidateAxesUpdate)
            {}

    __device__
    void operator()(const int& idx){
      
      for(uint i=0; i<numPolarizationAxes; i++){

        d3 candidateAxesUpdate = cellCandidateAxesUpdate[NUMAXESmax * idx + i];
        double norm = length(candidateAxesUpdate);

        if(norm > .0){
          candidateAxesUpdate = cellCandidateAxes[NUMAXESmax * idx + i] + .05 * candidateAxesUpdate/norm;
          cellCandidateAxes[NUMAXESmax * idx + i] = candidateAxesUpdate / length(candidateAxesUpdate);
        }
        else{
          cellCandidateAxes[NUMAXESmax * idx + i] = d3(.0);
        }
      }

    } // end operator()

  }; // end functor

  /** This functor updates the epithelial AB polarization axes. */
  struct evaluateEpithelialApicoBasalPolarity
  {
    const uint*                   cellType;
    const uint*                   cellTopologicalNeighbNum;
    const uint*                   cellTopologicalNeighbId;
    const d3*                     cellPosition;
    const uint*                   cellNeighbIsLateral;
    uint*                         cellEpiIsPolarized;
    const uint*                   cellEpiId;
    d3*                           cellAxisAB;
    uint*                         errorCode;
    const uint                    timer;
    
    evaluateEpithelialApicoBasalPolarity(
                uint*                   _cellType,
                uint*                   _cellTopologicalNeighbNum,
                uint*                   _cellTopologicalNeighbId,
                d3*                     _cellPosition,
                uint*                   _cellNeighbIsLateral,
                uint*                   _cellEpiIsPolarized,
                uint*                   _cellEpiId,
                d3*                     _cellAxisAB,
                uint*                   _errorCode,
                uint                    _timer
            )
         :
            cellType(_cellType),
            cellTopologicalNeighbNum(_cellTopologicalNeighbNum),
            cellTopologicalNeighbId(_cellTopologicalNeighbId),
            cellPosition(_cellPosition),
            cellNeighbIsLateral(_cellNeighbIsLateral),
            cellEpiIsPolarized(_cellEpiIsPolarized),
            cellEpiId(_cellEpiId),
            cellAxisAB(_cellAxisAB),
            errorCode(_errorCode),
            timer(_timer)
            {}

    __device__
    void operator()(const int& idx){
      
      uint celltype = cellType[idx];
      uint cellEpiPola = cellEpiIsPolarized[idx];

      // Undifferentiated and mesenchymal cells are not concerned by this function
      if(!(celltype == 2 && cellEpiPola)){return;}
      
      uint cellEpiId1 = cellEpiId[idx];

      // *******************************************
      // ****** Current Axis / candidate axis ******
      // *******************************************

      d3 axis = cellAxisAB[idx];

      //compute two vectors orthogonal to axis
      d3 normal2(-axis.y, axis.x,0);
      normal2 /= length(normal2);
      d3 normal3 = cross(axis, normal2);


      double angleArray[NUMNEIGHBTOPOmax];
      uint   rankArray[NUMNEIGHBTOPOmax];

      uint numNeighb  = cellTopologicalNeighbNum[idx], numNeighbEpi = 0, numNeighbEpiPola = 0;
      uint numValidTriangle = 0;
      d3 pos = cellPosition[idx], relPos;
      d3 vectNeighbPola(.0);

      for(uint i=0; i<numNeighb; i++){
        
        uint topoNeighbIndex  = idx * NUMNEIGHBTOPOmax + i;
        uint neighbCellId     = cellTopologicalNeighbId[topoNeighbIndex];
        uint celltype2        = cellType[neighbCellId];
        
        // Only lateral epithelial neighbors are taken into accound
        if( celltype2 >= 2  && (cellEpiId1 == cellEpiId[neighbCellId]) ){  

          // compute angle for epithelium apico-basal axis
          relPos = cellPosition[neighbCellId] - pos;
          relPos /= length(relPos);

          // Projection sur n2 et n3
          double p2 = dot(relPos, normal2);
          double p3 = dot(relPos, normal3);
          
          angleArray[numNeighbEpi] = acos( p2 / (sqrt(p2*p2+p3*p3)+.00001) );
          
          // angle must be between 0 and 2 times pi
          if( p3 < 0){
            angleArray[numNeighbEpi] = PI2 - angleArray[numNeighbEpi];
          }

          rankArray[numNeighbEpi] = neighbCellId;

          numNeighbEpi++;

          if(cellEpiIsPolarized[neighbCellId] == 1){
            vectNeighbPola += cellAxisAB[neighbCellId];
            numNeighbEpiPola++;
          }
        }

      }

      if(numNeighbEpi==0){return;}

      // Sorting neighbors by angle : insertion sort from http://www.concentric.net/~ttwang/sort/sort.htm
      double prev_val = angleArray[0], cur_val, temp_val;
        double prev_val_R = rankArray[0], cur_val_R, temp_val_R;
      uint indx2;
      
      for (uint indx = 1; indx < numNeighbEpi; ++indx){
        
        cur_val = angleArray[indx];
          cur_val_R = rankArray[indx];
        
        if ( prev_val > cur_val ){
        
          angleArray[indx] = prev_val; // move up the larger item first 
          rankArray[indx] = prev_val_R; // move up the larger item first 

          // find the insertion point for the smaller item 
          for (indx2 = indx - 1; indx2 > 0;){
            
          temp_val = angleArray[indx2 - 1];
            temp_val_R = rankArray[indx2 - 1];
          
          if ( temp_val > cur_val )
          {
            angleArray[indx2] = temp_val;
            rankArray[indx2] = temp_val_R;
            // still out of order, move up 1 slot to make room 
            indx2--;
          }
          else
            break;
          }
          angleArray[indx2] = cur_val; // insert the smaller item right here 
          rankArray[indx2] = cur_val_R; // insert the smaller item right here 
          
        }
        else{
          // in order, advance to next element 
          prev_val = cur_val;
          prev_val_R = cur_val_R;
        }
      }


      d3 AB, AC = cellPosition[ rankArray[0] ] - pos;
      d3 normalnei;
      d3 normal(.0);
      double diff;

      for(uint i=1; i<numNeighbEpi; i++){

        diff = (angleArray[i] - angleArray[i - 1]);
        
        if(diff < 0){
          diff += PI2;
        } 
        
        AB = AC;
        AC = cellPosition[rankArray[i]] - pos;

        //add only the normal contribution if the angle BAC is less than PI, 
        // otherwise, we consider the neighbors to be on a border
        // more strict angle limit could be PI3ov4 or PIov2
        if( diff < PI ){
          normalnei = cross(AB, AC);          
          normalnei /= length(normalnei);
          normal += normalnei;
          numValidTriangle++;
        }
      }

      diff = (angleArray[0] - angleArray[numNeighbEpi - 1]);
      
      if(diff < 0){
        diff += PI2;
      }
      
      if( diff < PI ){
        AB = cellPosition[rankArray[numNeighbEpi - 1]] - pos;
        AC = cellPosition[rankArray[0]] - pos;
        normalnei = cross(AB, AC);
        normalnei /= length(normalnei);
        normal += normalnei;
        numValidTriangle++;
      }
      
      double dist = length(normal);
      
      // Update ABaxis iff lateral reinforcement
      // Otherwise, consider epi unpolarized
      
      if(numValidTriangle == 0){
        cellEpiIsPolarized[idx] = 0;
      }
      else{
      
        if(numValidTriangle > 0 && dist != .0){
          normal /= dist;
          //update axis with inertia
          axis += .05 * normal;
        }
        
        axis /= length(axis);
        cellAxisAB[idx] = axis;
      }

    } // end operator()

  }; // end functor

} //end namespace

#endif
