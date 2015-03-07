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

#ifndef _CUSTOMSPANEIGHBFUNCTORS_H
#define _CUSTOMSPANEIGHBFUNCTORS_H

#include "spatialNeighborhood.hpp"

namespace mg {

  struct custom_yolk_metric_neighborhood
  {
      const d3*       yolkPosition;
      const double    gridBoxSize;
      const double    worldOrigin;
      const uint      gridSize;
      const uint*     gridPartNum;
      const uint*     gridPartId;
      uint*           yolkInteriorMetricNeighbNum;
      uint*           yolkInteriorMetricNeighbId;
      uint*           errorCode;

      custom_yolk_metric_neighborhood(
                  d3*         _yolkPosition,
                  double      _gridBoxSize,
                  double      _worldOrigin,
                  uint        _gridSize,
                  uint*       _gridPartNum,
                  uint*       _gridPartId,
                  uint*       _yolkInteriorMetricNeighbNum,
                  uint*       _yolkInteriorMetricNeighbId,
                  uint*       _errorCode
              )
            :
              yolkPosition(_yolkPosition),
              gridBoxSize(_gridBoxSize),
              worldOrigin(_worldOrigin),
              gridSize(_gridSize),
              gridPartNum(_gridPartNum),
              gridPartId(_gridPartId),
              yolkInteriorMetricNeighbNum(_yolkInteriorMetricNeighbNum),
              yolkInteriorMetricNeighbId(_yolkInteriorMetricNeighbId),
              errorCode(_errorCode)
              {}

      __device__
      void operator()(const int& idx){

        // printf("customspatial %d\n",idx);

        d3 pos = yolkPosition[idx], pos2, relPos;

        u3 gridPos = calcGridPos(pos, gridBoxSize, worldOrigin), neighGridPos;

        uint gridHash, neighbId, numNeighb = 0;
        double dist;

        // printf("cutoff dist %lf\n", gridBoxSize);

        i3 posMax(1,1,1);
        i3 posMin(-1,-1,-1);
    
        if(gridPos.x == 0){ posMin.x = 0; }
        if(gridPos.x == gridSize - 1){ posMax.x = 0; }

        if(gridPos.y == 0){ posMin.y = 0; }
        if(gridPos.y == gridSize - 1){ posMax.y = 0; }

        if(gridPos.z == 0){ posMin.z = 0; }
        if(gridPos.z == gridSize - 1){ posMax.z = 0; }

        for(int z=posMin.z; z<=posMax.z; z++) {
          for(int y=posMin.y; y<=posMax.y; y++) {
            for(int x=posMin.x; x<=posMax.x; x++) {

              neighGridPos = gridPos + u3(x, y, z);
              gridHash = calcGridHash(neighGridPos, gridSize);
              
              for(uint i=0;i<gridPartNum[gridHash];i++){
                  
                neighbId = gridPartId[ YOLKGRIDNUMPARTmax * gridHash + i];
                
                if (neighbId != idx){
        
                  pos2 = yolkPosition[neighbId];
                  relPos = pos2 - pos;
                  dist = length(relPos);

                  if (dist < gridBoxSize) {

                    // printf("   neighb %d num %d\n", neighbId, numNeighb);
                    yolkInteriorMetricNeighbId[(idx-NUMPARTYOLKMEMBRANE) * NUMNEIGHBMETRICmax + numNeighb] = neighbId;
                    numNeighb++;
                  } 
                    
                }
              }
            }
          }
        }

        if(numNeighb >= NUMNEIGHBMETRICmax){
          printf("error: too many yolk interior particle neighbor.\n");
          errorCode[0] = 102;
          return;
        }

        yolkInteriorMetricNeighbNum[idx-NUMPARTYOLKMEMBRANE] = numNeighb;

      } // end operator()

  }; // end functor custom_yolk_metric_neighborhood

  struct custom_yolk_topological_neighborhood
  {
      const d3*       yolkPosition;
      const uint*     yolkInteriorMetricNeighbNum;
      const uint*     yolkInteriorMetricNeighbId;
      uint*           yolkInteriorTopologicalNeighbNum;
      uint*           yolkInteriorTopologicalNeighbId;
      uint*           errorCode;

      custom_yolk_topological_neighborhood(
                  d3*         _yolkPosition,
                  uint*       _yolkInteriorMetricNeighbNum,
                  uint*       _yolkInteriorMetricNeighbId,
                  uint*       _yolkInteriorTopologicalNeighbNum,
                  uint*       _yolkInteriorTopologicalNeighbId,
                  uint*       _errorCode
              )
            :
              yolkPosition(_yolkPosition),
              yolkInteriorMetricNeighbNum(_yolkInteriorMetricNeighbNum),
              yolkInteriorMetricNeighbId(_yolkInteriorMetricNeighbId),
              yolkInteriorTopologicalNeighbNum(_yolkInteriorTopologicalNeighbNum),
              yolkInteriorTopologicalNeighbId(_yolkInteriorTopologicalNeighbId),
              errorCode(_errorCode)
              {}

      __device__
      void operator()(const int& idx){

        uint idx_interior = idx-NUMPARTYOLKMEMBRANE;
        uint numNeighb = 0;
        d3 pos = yolkPosition[idx];

        for(uint i=0;i<yolkInteriorMetricNeighbNum[idx_interior];i++){
          
          uint neighbId = yolkInteriorMetricNeighbId[(idx_interior) * NUMNEIGHBMETRICmax + i];
          d3 center = .5*(yolkPosition[neighbId] + pos);
          double dist = length(center-pos);

          uint keep = 1;
          
          for(uint j=0;j<yolkInteriorMetricNeighbNum[idx_interior];j++){
            
            uint neighbNeighbId = yolkInteriorMetricNeighbId[(idx_interior) * NUMNEIGHBMETRICmax + j];
            d3 neighbNeighbPos = yolkPosition[neighbNeighbId];

            uint test = (neighbId!=neighbNeighbId) && ( length(neighbNeighbPos - center) < dist );

            if(test)
            {
              keep = 0;
            }
            if(test)
            {
              break;
            }
          }
          
          if(keep){
            yolkInteriorTopologicalNeighbId[(idx_interior) * NUMNEIGHBTOPOmax + numNeighb] = neighbId;
            numNeighb++;
          }
        }

        if(numNeighb >= NUMNEIGHBTOPOmax){
          errorCode[0] = 103;
          return;
        }

        yolkInteriorTopologicalNeighbNum[idx_interior] = numNeighb;

      } // end operator()

  }; // end functor custom_yolk_topological_neighborhood

  struct custom_evl_metric_neighborhood
  {
      const d3*       evlPosition;
      const double    gridBoxSize;
      const double    worldOrigin;
      const uint      gridSize;
      const uint*     gridPartNum;
      const uint*     gridPartId;
      uint*           evlMetricNeighbNum;
      uint*           evlMetricNeighbId;
      uint*           errorCode;

      custom_evl_metric_neighborhood(
                  d3*         _evlPosition,
                  double      _gridBoxSize,
                  double      _worldOrigin,
                  uint        _gridSize,
                  uint*       _gridPartNum,
                  uint*       _gridPartId,
                  uint*       _evlMetricNeighbNum,
                  uint*       _evlMetricNeighbId,
                  uint*       _errorCode
              )
            :
              evlPosition(_evlPosition),
              gridBoxSize(_gridBoxSize),
              worldOrigin(_worldOrigin),
              gridSize(_gridSize),
              gridPartNum(_gridPartNum),
              gridPartId(_gridPartId),
              evlMetricNeighbNum(_evlMetricNeighbNum),
              evlMetricNeighbId(_evlMetricNeighbId),
              errorCode(_errorCode)
              {}

      __device__
      void operator()(const int& idx){

        d3 pos = evlPosition[idx], pos2, relPos;

        u3 gridPos = calcGridPos(pos, gridBoxSize, worldOrigin), neighGridPos;

        uint gridHash, neighbId, numNeighb = 0;
        double dist;

        i3 posMax(1,1,1);
        i3 posMin(-1,-1,-1);
    
        if(gridPos.x == 0){ posMin.x = 0; }
        if(gridPos.x == gridSize - 1){ posMax.x = 0; }

        if(gridPos.y == 0){ posMin.y = 0; }
        if(gridPos.y == gridSize - 1){ posMax.y = 0; }

        if(gridPos.z == 0){ posMin.z = 0; }
        if(gridPos.z == gridSize - 1){ posMax.z = 0; }

        for(int z=posMin.z; z<=posMax.z; z++) {
          for(int y=posMin.y; y<=posMax.y; y++) {
            for(int x=posMin.x; x<=posMax.x; x++) {

              neighGridPos = gridPos + u3(x, y, z);
              gridHash = calcGridHash(neighGridPos, gridSize);
              
              for(uint i=0;i<gridPartNum[gridHash];i++){
                  
                neighbId = gridPartId[ EVLGRIDNUMPARTmax * gridHash + i];
                
                if (neighbId != idx){
        
                  pos2 = evlPosition[neighbId];
                  relPos = pos2 - pos;
                  dist = length(relPos);

                  if (dist < gridBoxSize) {
                    evlMetricNeighbId[idx * NUMNEIGHBMETRICmax + numNeighb] = neighbId;
                    numNeighb++;
                  } 
                }
              }
            }
          }
        }

        if(numNeighb >= NUMNEIGHBMETRICmax){
          printf("error: too many evl particle neighbor.\n");
          errorCode[0] = 104;
          return;
        }

        evlMetricNeighbNum[idx] = numNeighb;

      } // end operator()

  }; // end functor custom_evl_metric_neighborhood

  struct custom_evl_topological_neighborhood
  {
      const d3*       evlPosition;
      const uint*     evlMetricNeighbNum;
      const uint*     evlMetricNeighbId;
      uint*           evlTopologicalNeighbNum;
      uint*           evlTopologicalNeighbId;
      uint*           errorCode;

      custom_evl_topological_neighborhood(
                  d3*         _evlPosition,
                  uint*       _evlMetricNeighbNum,
                  uint*       _evlMetricNeighbId,
                  uint*       _evlTopologicalNeighbNum,
                  uint*       _evlTopologicalNeighbId,
                  uint*       _errorCode
              )
            :
              evlPosition(_evlPosition),
              evlMetricNeighbNum(_evlMetricNeighbNum),
              evlMetricNeighbId(_evlMetricNeighbId),
              evlTopologicalNeighbNum(_evlTopologicalNeighbNum),
              evlTopologicalNeighbId(_evlTopologicalNeighbId),
              errorCode(_errorCode)
              {}

      __device__
      void operator()(const int& idx){

        uint numNeighb = 0;
        d3 pos = evlPosition[idx];

        for(uint i=0;i<evlMetricNeighbNum[idx];i++){
          
          uint neighbId = evlMetricNeighbId[idx * NUMNEIGHBMETRICmax + i];
          d3 center = .5*(evlPosition[neighbId] + pos);
          double dist = length(center-pos);

          uint keep = 1;
          
          for(uint j=0;j<evlMetricNeighbNum[idx];j++){
            
            uint neighbNeighbId = evlMetricNeighbId[idx * NUMNEIGHBMETRICmax + j];
            d3 neighbNeighbPos = evlPosition[neighbNeighbId];

            uint test = (neighbId!=neighbNeighbId) && ( length(neighbNeighbPos - center) < dist );

            if(test)
            {
              keep = 0;
            }
            if(test)
            {
              break;
            }
          }
          
          if(keep){
            evlTopologicalNeighbId[idx * NUMNEIGHBTOPOmax + numNeighb] = neighbId;
            numNeighb++;
          }
        }

        if(numNeighb >= NUMNEIGHBTOPOmax){
          errorCode[0] = 105;
          return;
        }

        evlTopologicalNeighbNum[idx] = numNeighb;

      } // end operator()

  }; // end functor custom_evl_topological_neighborhood

  struct custom_cellsyolk_metric_neighborhood
  {
      const d3*       yolkPosition;
      const d3*       cellPosition;
      const d3*       cellRadius;
      const double    cmax;
      const double    yolkMembraneRadius;
      const double    gridBoxSize;
      const double    worldOrigin;
      const uint      gridSize;
      const uint*     gridPartNum;
      const uint*     gridPartId;
      const uint*     yolkMembraneActivated;
      uint*           yolkCellsNeighbNum;
      uint*           yolkCellsNeighbId;
      uint*           cellsYolkNeighbNum;
      uint*           errorCode;

      custom_cellsyolk_metric_neighborhood(
                  d3*         _yolkPosition,
                  d3*         _cellPosition,
                  d3*         _cellRadius,
                  double      _cmax,
                  double      _yolkMembraneRadius,
                  double      _gridBoxSize,
                  double      _worldOrigin,
                  uint        _gridSize,
                  uint*       _gridPartNum,
                  uint*       _gridPartId,
                  uint*       _yolkMembraneActivated,
                  uint*       _yolkCellsNeighbNum,
                  uint*       _yolkCellsNeighbId,
                  uint*       _cellsYolkNeighbNum,
                  uint*       _errorCode
              )
            :
              yolkPosition(_yolkPosition),
              cellPosition(_cellPosition),
              cellRadius(_cellRadius),
              cmax(_cmax),
              yolkMembraneRadius(_yolkMembraneRadius),
              gridBoxSize(_gridBoxSize),
              worldOrigin(_worldOrigin),
              gridSize(_gridSize),
              gridPartNum(_gridPartNum),
              gridPartId(_gridPartId),
              yolkMembraneActivated(_yolkMembraneActivated),
              yolkCellsNeighbNum(_yolkCellsNeighbNum),
              yolkCellsNeighbId(_yolkCellsNeighbId),
              cellsYolkNeighbNum(_cellsYolkNeighbNum),
              errorCode(_errorCode)
              {}

      __device__
      void operator()(const int& idx){

        int imin = -1;

        if(yolkMembraneActivated[idx] == 1){

          d3 pos = yolkPosition[idx];

          u3 gridPos = calcGridPos(pos, gridBoxSize, worldOrigin);
          
          double distmin = gridBoxSize;

          i3 posMax(1,1,1);
          i3 posMin(-1,-1,-1);
      
          if(gridPos.x == 0){ posMin.x = 0; }
          if(gridPos.x == gridSize - 1){ posMax.x = 0; }

          if(gridPos.y == 0){ posMin.y = 0; }
          if(gridPos.y == gridSize - 1){ posMax.y = 0; }

          if(gridPos.z == 0){ posMin.z = 0; }
          if(gridPos.z == gridSize - 1){ posMax.z = 0; }

          for(int z=posMin.z; z<=posMax.z; z++) {
            for(int y=posMin.y; y<=posMax.y; y++) {
              for(int x=posMin.x; x<=posMax.x; x++) {

                u3 neighGridPos = gridPos + u3(x, y, z);
                uint gridHash = calcGridHash(neighGridPos, gridSize);
                
                for(uint i=0;i<gridPartNum[gridHash];i++){
                    
                  uint neighbId = gridPartId[ CELLSYOLKGRIDNUMPARTmax * gridHash + i];
                  
                  d3 pos2 = cellPosition[neighbId];
                  d3 relPos = pos2 - pos;
                  double dist = length(relPos);

                  if(dist < distmin){
                    double radius2 = cellRadius[neighbId].x;

                    if (dist < cmax * (radius2+yolkMembraneRadius) ) {
                      distmin = dist;
                      imin = neighbId;
                    } 

                  }
                }
              }
            }
          }
        }

        uint numNeighb = 0;

        if(imin != -1){
          yolkCellsNeighbId[idx] = imin;
          numNeighb++;

          cellsYolkNeighbNum[imin] = 1;
        }

        yolkCellsNeighbNum[idx] = numNeighb;


      } // end operator()

  }; // end functor custom_cellsyolk_metric_neighborhood

  struct custom_cellsevl_neighborhood
  {
      const d3*       evlPosition;
      const d3*       evlNormal;
      const d3*       cellPosition;
      const d3*       cellRadius;
      const double    evlRadiusAB;
      const double    gridBoxSize;
      const double    worldOrigin;
      const uint      gridSize;
      const uint*     gridPartNum;
      const uint*     gridPartId;
      const uint*     cellTopologicalNeighbNum;
      const uint*     cellTopologicalNeighbId;
      uint*           cellsEvlNeighbNum;
      uint*           cellsEvlNeighbId;
      uint*           errorCode;

      custom_cellsevl_neighborhood(
                  d3*         _evlPosition,
                  d3*         _evlNormal,
                  d3*         _cellPosition,
                  d3*         _cellRadius,
                  double      _evlRadiusAB,
                  double      _gridBoxSize,
                  double      _worldOrigin,
                  uint        _gridSize,
                  uint*       _gridPartNum,
                  uint*       _gridPartId,
                  uint*       _cellTopologicalNeighbNum,
                  uint*       _cellTopologicalNeighbId,
                  uint*       _cellsEvlNeighbNum,
                  uint*       _cellsEvlNeighbId,
                  uint*       _errorCode
              )
            :
              evlPosition(_evlPosition),
              evlNormal(_evlNormal),
              cellPosition(_cellPosition),
              cellRadius(_cellRadius),
              evlRadiusAB(_evlRadiusAB),
              gridBoxSize(_gridBoxSize),
              worldOrigin(_worldOrigin),
              gridSize(_gridSize),
              gridPartNum(_gridPartNum),
              gridPartId(_gridPartId),
              cellTopologicalNeighbNum(_cellTopologicalNeighbNum),
              cellTopologicalNeighbId(_cellTopologicalNeighbId),
              cellsEvlNeighbNum(_cellsEvlNeighbNum),
              cellsEvlNeighbId(_cellsEvlNeighbId),
              errorCode(_errorCode)
              {}

      __device__
      void operator()(const int& idx){

        d3 pos = cellPosition[idx];
        double radius = cellRadius[idx].x;

        u3 gridPos = calcGridPos(pos, gridBoxSize, worldOrigin);
        
        double distmin = gridBoxSize;
        int imin = -1;
        uint numNeighb = 0;

        i3 posMax(1,1,1);
        i3 posMin(-1,-1,-1);
    
        if(gridPos.x == 0){ posMin.x = 0; }
        if(gridPos.x == gridSize - 1){ posMax.x = 0; }

        if(gridPos.y == 0){ posMin.y = 0; }
        if(gridPos.y == gridSize - 1){ posMax.y = 0; }

        if(gridPos.z == 0){ posMin.z = 0; }
        if(gridPos.z == gridSize - 1){ posMax.z = 0; }

        for(int z=posMin.z; z<=posMax.z; z++) {
          for(int y=posMin.y; y<=posMax.y; y++) {
            for(int x=posMin.x; x<=posMax.x; x++) {

              u3 neighGridPos = gridPos + u3(x, y, z);
              uint gridHash = calcGridHash(neighGridPos, gridSize);
              
              for(uint i=0;i<gridPartNum[gridHash];i++){
                  
                uint neighbId = gridPartId[ EVLGRIDNUMPARTmax * gridHash + i];
                
                d3 pos2 = evlPosition[neighbId];
                d3 relPos = pos2 - pos;
                double dist = length(relPos);

                d3 normal2 = evlNormal[neighbId];
                double distN = dot(relPos, normal2);
                
                if(fabs(distN) < 1.5 * (radius+evlRadiusAB) && dist < distmin){
                  distmin = dist;
                  imin = neighbId;
                }
              }
            }
          }
        }

        // if a closest evl particle is found, it is kept as neighbor if no neighbor cell is located in between.
        if(imin != -1){

          d3 normal = evlNormal[imin];
          uint keep = 1;

          for(uint i=0; i<cellTopologicalNeighbNum[idx]; i++){
            uint topoNeighbIndex  = idx * NUMNEIGHBTOPOmax + i;
            uint neighbCellId     = cellTopologicalNeighbId[topoNeighbIndex];
            d3 relPos = cellPosition[neighbCellId] - pos;
            double dist = length(relPos);

            if( dot(relPos, normal) > dist * .75 ){
              keep = 0;
              break;
            }
          }

          if(keep){
            cellsEvlNeighbId[idx] = imin;
            numNeighb++;
          }
        }
        cellsEvlNeighbNum[idx] = numNeighb;
      } // end operator()

  }; // end functor custom_cellsyolk_neighborhood

  struct custom_yolkmarginevl_metric_neighborhood
  {
      const d3*       evlPosition;
      const d3*       yolkPosition;
      const d3*       evlRadius;
      const double    gridBoxSize;
      const double    worldOrigin;
      const uint      gridSize;
      const uint*     gridPartNum;
      const uint*     gridPartId;
      const uint*     yolkMembraneEYSL;
      uint*           yolkMarginEvlMetricNeighbNum;
      uint*           yolkMarginEvlMetricNeighbId;
      uint*           errorCode;

      custom_yolkmarginevl_metric_neighborhood(
                  d3*         _evlPosition,
                  d3*         _yolkPosition,
                  d3*         _evlRadius,
                  double      _gridBoxSize,
                  double      _worldOrigin,
                  uint        _gridSize,
                  uint*       _gridPartNum,
                  uint*       _gridPartId,
                  uint*       _yolkMembraneEYSL,
                  uint*       _yolkMarginEvlMetricNeighbNum,
                  uint*       _yolkMarginEvlMetricNeighbId,
                  uint*       _errorCode
              )
            :
              evlPosition(_evlPosition),
              yolkPosition(_yolkPosition),
              evlRadius(_evlRadius),
              gridBoxSize(_gridBoxSize),
              worldOrigin(_worldOrigin),
              gridSize(_gridSize),
              gridPartNum(_gridPartNum),
              gridPartId(_gridPartId),
              yolkMembraneEYSL(_yolkMembraneEYSL),
              yolkMarginEvlMetricNeighbNum(_yolkMarginEvlMetricNeighbNum),
              yolkMarginEvlMetricNeighbId(_yolkMarginEvlMetricNeighbId),
              errorCode(_errorCode)
              {}

      __device__
      void operator()(const int& idx){

        if(yolkMembraneEYSL[idx] != 1){
          yolkMarginEvlMetricNeighbNum[idx] = 0;
          return;
        }

        d3 pos = yolkPosition[idx];

        u3 gridPos = calcGridPos(pos, gridBoxSize, worldOrigin);
        
        uint numNeighb = 0;

        i3 posMax(1,1,1);
        i3 posMin(-1,-1,-1);
    
        if(gridPos.x == 0){ posMin.x = 0; }
        if(gridPos.x == gridSize - 1){ posMax.x = 0; }

        if(gridPos.y == 0){ posMin.y = 0; }
        if(gridPos.y == gridSize - 1){ posMax.y = 0; }

        if(gridPos.z == 0){ posMin.z = 0; }
        if(gridPos.z == gridSize - 1){ posMax.z = 0; }

        for(int z=posMin.z; z<=posMax.z; z++) {
          for(int y=posMin.y; y<=posMax.y; y++) {
            for(int x=posMin.x; x<=posMax.x; x++) {

              u3 neighGridPos = gridPos + u3(x, y, z);
              uint gridHash = calcGridHash(neighGridPos, gridSize);
              
              for(uint i=0;i<gridPartNum[gridHash];i++){
                  
                uint neighbId = gridPartId[ EVLGRIDNUMPARTmax * gridHash + i];
                
                d3 pos2 = evlPosition[neighbId];
                d3 relPos = pos2 - pos;
                double dist = length(relPos);

                double radius2 = evlRadius[neighbId].x;

                if(dist < 4.0 * radius2){
                  yolkMarginEvlMetricNeighbId[idx*NUMNEIGHBMETRICmax+numNeighb] = neighbId;
                  numNeighb++;
                }
              }
            }
          }
        }

        yolkMarginEvlMetricNeighbNum[idx] = numNeighb;
      } // end operator()

  }; // end functor custom_yolkmarginevl_metric_neighborhood

  struct custom_yolkmarginevl_topological_neighborhood
  {
      const d3*       evlPosition;
      const d3*       yolkPosition;
      const uint*     yolkMarginEvlMetricNeighbNum;
      const uint*     yolkMarginEvlMetricNeighbId;
      uint*           yolkMarginEvlTopologicalNeighbNum;
      uint*           yolkMarginEvlTopologicalNeighbId;
      uint*           errorCode;

      custom_yolkmarginevl_topological_neighborhood(
                  d3*         _evlPosition,
                  d3*         _yolkPosition,
                  uint*       _yolkMarginEvlMetricNeighbNum,
                  uint*       _yolkMarginEvlMetricNeighbId,
                  uint*       _yolkMarginEvlTopologicalNeighbNum,
                  uint*       _yolkMarginEvlTopologicalNeighbId,
                  uint*       _errorCode
              )
            :
              evlPosition(_evlPosition),
              yolkPosition(_yolkPosition),
              yolkMarginEvlMetricNeighbNum(_yolkMarginEvlMetricNeighbNum),
              yolkMarginEvlMetricNeighbId(_yolkMarginEvlMetricNeighbId),
              yolkMarginEvlTopologicalNeighbNum(_yolkMarginEvlTopologicalNeighbNum),
              yolkMarginEvlTopologicalNeighbId(_yolkMarginEvlTopologicalNeighbId),
              errorCode(_errorCode)
              {}

      __device__
      void operator()(const int& idx){

        uint numNeighb = 0;
        d3 pos = yolkPosition[idx];

        for(uint i=0;i<yolkMarginEvlMetricNeighbNum[idx];i++){
          
          uint neighbId = yolkMarginEvlMetricNeighbId[idx * NUMNEIGHBMETRICmax + i];
          d3 center = .5*(evlPosition[neighbId] + pos);
          double dist = length(center-pos);

          uint keep = 1;
          
          for(uint j=0;j<yolkMarginEvlMetricNeighbNum[idx];j++){
            
            uint neighbNeighbId = yolkMarginEvlMetricNeighbId[idx * NUMNEIGHBMETRICmax + j];
            d3 neighbNeighbPos = evlPosition[neighbNeighbId];

            uint test = (neighbId!=neighbNeighbId) && ( length(neighbNeighbPos - center) < dist );

            if(test)
            {
              keep = 0;
            }
            if(test)
            {
              break;
            }
          }
          
          if(keep){
            yolkMarginEvlTopologicalNeighbId[idx * NUMNEIGHBTOPOmax + numNeighb] = neighbId;
            numNeighb++;
          }
        }

        if(numNeighb >= NUMNEIGHBTOPOmax){
          errorCode[0] = 105;
          return;
        }

        yolkMarginEvlTopologicalNeighbNum[idx] = numNeighb;

      } // end operator()

  }; // end functor custom_yolkmarginevl_topological_neighborhood


}

#endif