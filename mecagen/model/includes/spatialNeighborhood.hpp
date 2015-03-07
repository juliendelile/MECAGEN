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

#ifndef _SPANEIGHBFUNCTORS_H
#define _SPANEIGHBFUNCTORS_H

#include "define.hpp"

#include "param_grn.hpp"

#include <cstring>
#include <stdio.h>	//printf
#include <assert.h>

namespace mg {

	// From thrust bounding box example
	// https://github.com/thrust/thrust/blob/master/examples/bounding_box.cu

	// bounding box type
	struct boundingBox
	{
	  __host__ __device__
	  boundingBox() {}

	  __host__ __device__
	  boundingBox(const d3 &position)
	    : lower_left(position), upper_right(position)
	  {}

	  __host__ __device__
	  boundingBox(const d3 &ll, const d3 &ur)
	    : lower_left(ll), upper_right(ur)
	  {}

	  d3 lower_left, upper_right;
	};

	// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
	struct bbox_reduction : public thrust::binary_function<boundingBox,boundingBox,boundingBox>
	{
	  __host__ __device__
	  boundingBox operator()(boundingBox a, boundingBox b)
	  {
	    // lower left corner
	    d3 ll(
	                    thrust::min(a.lower_left.x, b.lower_left.x),
	                    thrust::min(a.lower_left.y, b.lower_left.y),
	                    thrust::min(a.lower_left.z, b.lower_left.z)
	                    );

	    // upper right corner
	    d3 ur(
	                    thrust::max(a.upper_right.x, b.upper_right.x),
	                    thrust::max(a.upper_right.y, b.upper_right.y),
	                    thrust::max(a.upper_right.z, b.upper_right.z)
	                    );

	    return boundingBox(ll, ur);
	  }
	};

	struct compare_first_radius_value
	  {
	    __host__ __device__
	    bool operator()(d3 lhs, d3 rhs)
	    {
	      return lhs.x < rhs.x;
	    }
	  };

	/** This function determines each particle's associated grid box coordinates.*/
  inline
	__host__ __device__ u3 calcGridPos(d3 p, double gridBoxSize, double worldOrigin)
	{
	    u3 gridPos;
	    gridPos.x = floor((p.x - worldOrigin) / gridBoxSize);
	    gridPos.y = floor((p.y - worldOrigin) / gridBoxSize);
	    gridPos.z = floor((p.z - worldOrigin) / gridBoxSize);
	    return gridPos;
	}

	/** This function determines the index of a grid box from spatial coordinates.*/
  inline
	__host__ __device__ uint calcGridHash(u3 gridPos, uint gridSize)
	{
	    return gridPos.z * gridSize * gridSize + gridPos.y * gridSize + gridPos.x;
	}

  /** This functor calculates each particle's grid box id and stores it. */
	struct fill_grid
	{
	    const d3*       cellPosition;
	    const double    gridBoxSize;
	    const double    worldOrigin;
	    const uint      gridSize;
	    uint*           gridPartNum;
	    uint*           gridPartId;
	    const uint      gridBox_NumPartMax;
	    uint* 					errorCode;
      const uint      timestep;

	    fill_grid(
	                d3*         _cellPosition,
	                double      _gridBoxSize,
	                double      _worldOrigin,
	                uint        _gridSize,
	                uint*       _gridPartNum,
	                uint*       _gridPartId,
	                uint        _gridBox_NumPartMax,
	                uint* 		  _errorCode,
                  uint        _timestep
	            )
	         :
	            cellPosition(_cellPosition),
	            gridBoxSize(_gridBoxSize),
	            worldOrigin(_worldOrigin),
	            gridSize(_gridSize),
	            gridPartNum(_gridPartNum),
	            gridPartId(_gridPartId),
	            gridBox_NumPartMax(_gridBox_NumPartMax),
	            errorCode(_errorCode),
              timestep(_timestep)
	            {}

	    __device__
	    void operator()(const int& idx){

	        d3 p = cellPosition[idx];
	        u3 gridPos = calcGridPos(p, gridBoxSize, worldOrigin);
	        uint hash = calcGridHash(gridPos, gridSize);
	        uint boxcounter = mgAtomicAddOne( &gridPartNum[hash]);    

	        if(boxcounter >= gridBox_NumPartMax){
	        	printf("Error: cell %d box %d %d %d (hash %d) contains %d (max %d)\n", idx, gridPos.x, gridPos.y, gridPos.z, hash, boxcounter, gridBox_NumPartMax);
	        	errorCode[0] = 2;
	        }

	        gridPartId[ gridBox_NumPartMax * hash + boxcounter ] = idx;
	    }
	};

  /** This functor calculates the metric neighborhood of each cell.*/
  struct metric_neighborhood
  {
      const int       numCells;
      const d3*       cellPosition;
      const uint*     cellType;
      const d3*       cellAxisAB;
      const d3*       cellRadius;
      const double    gridBoxSize;
      const double    worldOrigin;
      const uint      gridSize;
      const uint*     gridPartNum;
      const uint*     gridPartId;
      const uint      gridBox_NumPartMax;
      const uint      numNeighbMax;
      const uint      timer;
      const double*   cellShapeRatio;
      uint*           cellMetricNeighbId;
      uint*           cellMetricNeighbNum;
      double*         cellNeighbAngle;
      uint*       errorCode;
      const MechaParams* mechaParams;

      metric_neighborhood(
                  int         _numCells,
                  d3*         _cellPosition,
                  uint*       _cellType,
                  d3*         _cellAxis,
                  d3*         _cellRadius,
                  double      _gridBoxSize,
                  double      _worldOrigin,
                  uint        _gridSize,
                  uint*       _gridPartNum,
                  uint*       _gridPartId,
                  uint        _gridBox_NumPartMax,
                  uint        _numNeighbMax,
                  uint        _timer,
                  double*     _cellShapeRatio,
                  uint*       _cellMetricNeighbId,
                  uint*       _cellMetricNeighbNum,
                  double*     _cellNeighbAngle,
                  uint*       _errorCode,
                  MechaParams* _mechaParams
              )
           :
              numCells(_numCells),
              cellPosition(_cellPosition),
              cellType(_cellType), 
              cellAxisAB(_cellAxis),
              cellRadius(_cellRadius),
              gridBoxSize(_gridBoxSize),
              worldOrigin(_worldOrigin),
              gridSize(_gridSize),
              gridPartNum(_gridPartNum),
              gridPartId(_gridPartId),
              gridBox_NumPartMax(_gridBox_NumPartMax),  
              numNeighbMax(_numNeighbMax),
              timer(_timer),
              cellShapeRatio(_cellShapeRatio),
              cellMetricNeighbId(_cellMetricNeighbId),
              cellMetricNeighbNum(_cellMetricNeighbNum),
              cellNeighbAngle(_cellNeighbAngle),
              errorCode(_errorCode),
              mechaParams(_mechaParams)
              {}

      __device__
      void operator()(const int& idx){

        d3 pos = cellPosition[idx], pos2, relPos;
        uint celltype = cellType[idx], celltype2;
        double shapeRatio;

        d3 normale1, normale2; 

        u3 gridPos = calcGridPos(pos, gridBoxSize, worldOrigin), neighGridPos;

        uint gridHash, neighbId, numNeighb = 0;
        double dist;

        i3 posMax(1,1,1);
        i3 posMin(-1,-1,-1);
    
        double radius1, radius1lat = cellRadius[idx].x, radius1ab, radius2;

        // If the cell is epithelial, the apicobasal radius is also taken into account
        if( celltype == 2){
          radius1ab = cellRadius[idx].y;
          normale1 = cellAxisAB[idx];
          shapeRatio = cellShapeRatio[idx];
        }
        
        // Visit neighbor grid boxes
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
                  
                neighbId = gridPartId[ gridBox_NumPartMax * gridHash + i];
                
                if (neighbId != idx){
        
                  pos2 = cellPosition[neighbId];
                  relPos = pos2 - pos;
                  dist = length(relPos); 
                  relPos /= dist;
                  celltype2 = cellType[neighbId];

                  double cmax = .0;

                  cmax = mechaParams[0].maximumDistanceCoefficient[celltype * 4 + celltype2];
                  
                  // Every cell have a single radius...
                  if(celltype != 2){
                    radius1 = radius1lat;
                  }
                  // ... but epithelial cell with apicobasal polarization have two radii.
                  // The selected radius depends whether the neighbor is lateral or not.
                  else{
                    double scal = dot( relPos, normale1 );
                    if( fabs(scal) <= shapeRatio ){
                      radius1 = radius1lat;
                    }
                    else{
                      radius1 = radius1ab;
                    }
                  }
                  

                  // the same goes for the neighbor cell
                  if(celltype2 != 2){
                    radius2 = cellRadius[neighbId].x;
                  }
                  else{
                    double scal = dot( relPos, cellAxisAB[neighbId] );
                    if( fabs(scal) < cellShapeRatio[neighbId] ){
                      radius2 = cellRadius[neighbId].x;
                    }
                    else{
                      radius2 = cellRadius[neighbId].y;
                    } 
                  }

                  if (dist < cmax * (radius1+radius2) ) {
                      cellMetricNeighbId[idx * numNeighbMax + numNeighb] = neighbId;
                      numNeighb++;
                  } 
                    
                }
              }
            }
          }
        }

        if(numNeighb >= NUMNEIGHBMETRICmax){
          errorCode[0] = 6;
          return;
        }

        cellMetricNeighbNum[idx] = numNeighb;

      } // end operator()

  }; // end functor metric_neighborhood

  /** This functor calculates the topological neighborhood of each cell.*/
	struct new_topological_neighborhood
	{
    const uint*     cellType;
    const uint*     cellMetricNeighbId;
    const uint*     cellMetricNeighbNum;	    
    uint*     		cellTopologicalNeighbId;
    uint*     		cellTopologicalNeighbNum;
    const double*   cellNeighbAngle;
    const uint      numNeighbMax;
    const d3*       cellPosition;
    const d3*       cellRadius;
    d3*       		cellAxisAB;
    const uint 		timer;
    uint* 	 		errorCode;

    new_topological_neighborhood(
                uint*       _cellType,
                uint*       _cellMetricNeighbId,
                uint*       _cellMetricNeighbNum,
                uint*       _cellTopologicalNeighbId,
                uint*       _cellTopologicalNeighbNum,
                double*     _cellNeighbAngle,
                uint        _numNeighbMax,
                d3*         _cellPosition,
                d3*         _cellRadius,
                d3*         _cellAxis,
                uint 		_timer,
                uint* 		_errorCode
            )
         :
            cellType(_cellType),
            cellMetricNeighbId(_cellMetricNeighbId),
            cellMetricNeighbNum(_cellMetricNeighbNum),	            
            cellTopologicalNeighbId(_cellTopologicalNeighbId),
            cellTopologicalNeighbNum(_cellTopologicalNeighbNum),
            cellNeighbAngle(_cellNeighbAngle),
            numNeighbMax(_numNeighbMax),
            cellPosition(_cellPosition), 
            cellRadius(_cellRadius),
            cellAxisAB(_cellAxis),
            timer(_timer),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){

    	uint neighbId, neighbNeighbId, keep, test, numNeighb = 0;
    	d3 pos = cellPosition[idx], center, neighbNeighbPos;
    	double dist;

			for(uint i=0;i<cellMetricNeighbNum[idx];i++){
				
				neighbId = cellMetricNeighbId[idx * numNeighbMax + i];
				center = .5*(cellPosition[neighbId] + pos);
				dist = length(center-pos);

        keep = 1;
				
				for(uint j=0;j<cellMetricNeighbNum[idx];j++){
					
					neighbNeighbId = cellMetricNeighbId[idx * numNeighbMax + j];
					neighbNeighbPos = cellPosition[neighbNeighbId];

					test = (neighbId!=neighbNeighbId) && ( length(neighbNeighbPos - center) < dist );

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
					cellTopologicalNeighbId[idx * NUMNEIGHBTOPOmax + numNeighb] = neighbId;
					numNeighb++;
				}
			}

			if(numNeighb >= NUMNEIGHBTOPOmax){
				errorCode[0] = 3;
				return;
			}

			cellTopologicalNeighbNum[idx] = numNeighb;

    }// end operator()
	};// end functor new_topological_neighborhood

}

#endif