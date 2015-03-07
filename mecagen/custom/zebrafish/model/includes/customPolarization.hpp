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

#ifndef _CUSTOMPOLAFUNCTORS_H
#define _CUSTOMPOLAFUNCTORS_H

namespace mg {

  struct updateEVLNormales
  {
    const d3*                     evlPosition;
    d3*                           evlNormal;
    const uint*                   evlTopologicalNeighbNum;
    const uint*                   evlTopologicalNeighbId;
    uint*                         errorCode;
    
    updateEVLNormales(
                d3*                     _evlPosition,
                d3*                     _evlNormal,
                uint*                   _evlTopologicalNeighbNum,
                uint*                   _evlTopologicalNeighbId,
                uint*                   _errorCode
            )
         :
            evlPosition(_evlPosition),
            evlNormal(_evlNormal),
            evlTopologicalNeighbNum(_evlTopologicalNeighbNum),
            evlTopologicalNeighbId(_evlTopologicalNeighbId),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){
      
      d3 pos = evlPosition[idx];

      d3 normal = evlNormal[idx];
      d3 normal2(-normal.y, normal.x,0);
      normal2 /= length(normal2);
      d3 normal3 = cross(normal, normal2);

      double angleArray[NUMNEIGHBTOPOmax];
      uint   rankArray[NUMNEIGHBTOPOmax];

      uint numNeighb = evlTopologicalNeighbNum[idx];
      if(numNeighb==0){return;}

      for(uint i=0; i<numNeighb; i++){
        
        uint topoNeighbIndex  = idx * NUMNEIGHBTOPOmax + i;
        uint neighbCellId     = evlTopologicalNeighbId[topoNeighbIndex];

        d3 relPos = evlPosition[neighbCellId] - pos;
        relPos /= length(relPos);

        double p2 = dot(relPos, normal2);
        double p3 = dot(relPos, normal3);
        
        angleArray[i] = acos( p2 / (sqrt(p2*p2+p3*p3)+.00001) );
        
        // angle must be between 0 and 2 times pi
        if( p3 < 0){
          angleArray[i] = PI2 - angleArray[i];
        }

        rankArray[i] = neighbCellId;
      }

      // Sorting neighbors by angle : insertion sort from http://www.concentric.net/~ttwang/sort/sort.htm
      double prev_val = angleArray[0], cur_val, temp_val;
        double prev_val_R = rankArray[0], cur_val_R, temp_val_R;
      uint indx2;
      
      for (uint indx = 1; indx < numNeighb; ++indx){
        
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

      d3 AB, AC = evlPosition[ rankArray[0] ] - pos;
      d3 normalnei;
      d3 new_normal(.0);
      double diff;

      for(uint i=1; i<numNeighb; i++){

        diff = (angleArray[i] - angleArray[i - 1]);
        
        if(diff < 0){
          diff += PI2;
        } 
        
        AB = AC;
        AC = evlPosition[rankArray[i]] - pos;
        
        //add only the normal contribution if the angle BAC is less than PI, 
        // otherwise, we consider the neighbors to be on a border
        // more strict angle limit could be PI3ov4 or PIov2
        if( diff < PI ){
          normalnei = cross(AB, AC);          
          normalnei /= length(normalnei);
          new_normal += normalnei;
        }
      }

      diff = (angleArray[0] - angleArray[numNeighb - 1]);
      
      if(diff < 0){
        diff += PI2;
      }
      
      if( diff < PI ){
        AB = evlPosition[rankArray[numNeighb - 1]] - pos;
        AC = evlPosition[rankArray[0]] - pos;
        normalnei = cross(AB, AC);
        normalnei /= length(normalnei);
        new_normal += normalnei;
      }
      
      double dist = length(new_normal);
      
      if(dist != .0){
        evlNormal[idx] = new_normal / dist;
      }

    } // end operator()

  }; // end functor

}

#endif