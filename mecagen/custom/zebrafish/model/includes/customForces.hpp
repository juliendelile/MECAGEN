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

#ifndef _CUSTOMFORCEFUNCTORS_H
#define _CUSTOMFORCEFUNCTORS_H

#include "forces.hpp"
#include "cellbehavior.hpp"

namespace mg {

  struct custom_yolk_forces_computation
  {
    const d3*             yolkPosition;
    const double          yolkInteriorRadius;
    const CustomParams*   customParam;
    const double          yolkInteriorEquilibriumDistance;
    const uint*           yolkInteriorTopologicalNeighbNum;
    const uint*           yolkInteriorTopologicalNeighbId;
    const uint*           yolkMembraneNeighbNum;
    const uint*           yolkMembraneNeighbId;
    const double*         yolkMembraneNeighbRL;
    d3*                   yolkForces;
    uint*                 yolkForcesNum;
    uint*                 errorCode;

    custom_yolk_forces_computation(
                d3*           _yolkPosition,
                double        _yolkInteriorRadius,
                CustomParams* _customParam,
                double        _yolkInteriorEquilibriumDistance,
                uint*         _yolkInteriorTopologicalNeighbNum,
                uint*         _yolkInteriorTopologicalNeighbId,
                uint*         _yolkMembraneNeighbNum,
                uint*         _yolkMembraneNeighbId,
                double*       _yolkMembraneNeighbRL,
                d3*           _yolkForces,
                uint*         _yolkForcesNum,
                uint*         _errorCode
            )
          :
            yolkPosition(_yolkPosition),
            yolkInteriorRadius(_yolkInteriorRadius),
            customParam(_customParam),
            yolkInteriorEquilibriumDistance(_yolkInteriorEquilibriumDistance),
            yolkInteriorTopologicalNeighbNum(_yolkInteriorTopologicalNeighbNum),
            yolkInteriorTopologicalNeighbId(_yolkInteriorTopologicalNeighbId),
            yolkMembraneNeighbNum(_yolkMembraneNeighbNum),
            yolkMembraneNeighbId(_yolkMembraneNeighbId),
            yolkMembraneNeighbRL(_yolkMembraneNeighbRL),
            yolkForces(_yolkForces),
            yolkForcesNum(_yolkForcesNum),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){

      d3 pos1 = yolkPosition[idx];

      d3 f(0.0);

      if(idx < NUMPARTYOLKMEMBRANE){      // yolk membrane particles

        double yMStiffness = customParam[0].yolkMembraneStiffness;
        double yMRLCoeff = customParam[0].yolkMembraneRLCoeff;

        // Loop over all yolk membrane neighbors (rank 1 and rank 2)
        for(uint i=0; i < yolkMembraneNeighbNum[idx+NUMPARTYOLKMEMBRANE];i++){
          
          uint neighbIndex = idx*NUMNEIGHMYMYmax+i;
          uint neighbCellId = yolkMembraneNeighbId[neighbIndex];
          d3 relPos = yolkPosition[neighbCellId] - pos1;
          double dist = length(relPos);
          relPos /= dist;

          // null distance may happen if the yolk membrane gets coiled
          if(dist != 0){
            f-= yMStiffness * ( yMRLCoeff * yolkMembraneNeighbRL[neighbIndex] - dist ) * relPos;
          }
        }
        
      }
      else{                               // yolk interior particles

        uint idx_interior = idx-NUMPARTYOLKMEMBRANE;
        uint numNeighb  = yolkInteriorTopologicalNeighbNum[idx_interior];

        // Fix 1.1
        double yIRadius = yolkInteriorRadius;
        double yICmax   = customParam[0].yolkInteriorCmax;
        double yISurfaceScaling = customParam[0].yolkInteriorSurfaceScaling;
        double yIEquilibriumDistance = yolkInteriorEquilibriumDistance;
        double yIAttractionCoefficient = customParam[0].yolkInteriorAttractionCoefficient;
        double yIRepulsionCoefficient = customParam[0].yolkInteriorRepulsionCoefficient;
      
        d3 f_temp;

        for(uint i=0; i<numNeighb; i++){
        
          uint topoNeighbIndex    = idx_interior * NUMNEIGHBTOPOmax + i;
          uint neighbCellId       = yolkInteriorTopologicalNeighbId[topoNeighbIndex];
          d3 pos2                 = yolkPosition[neighbCellId];
          d3 relPos               = pos2 - pos1;
          double dist             = length(relPos);
          relPos /= dist;

          double surface = surface_estimation(
                                    &dist,
                                    &yIRadius,
                                    &yIRadius,
                                    &yICmax,
                                    &yISurfaceScaling
                                    );
          double disteq  = yIEquilibriumDistance;

          /***********************************
          **** Attraction-repulsion force ****
          ***********************************/
          //NEW MECAGEN
          f_temp = attr_rep_colinear_force2( 
                            &dist,
                            &disteq,
                            &surface,
                            &yIAttractionCoefficient,
                            &yIRepulsionCoefficient
                            ) * relPos;

          f += f_temp;

          /***********************************
          **** Transfer force on neighbor ****
          ***********************************/

          yolkForces[neighbCellId * NUMFORCEmax + mgAtomicAddOne(&yolkForcesNum[neighbCellId])] = -f_temp;
          
        }
      }

      yolkForces[idx * NUMFORCEmax + mgAtomicAddOne(&yolkForcesNum[idx])] = f; 
      
    } // end operator()
  }; // end functor custom_yolk_forces_computation

  struct custom_yolkcells_forces_computation
  {
    const d3*             yolkPosition;
    const d3*             cellPosition;
    const uint*           yolkMembraneActivated;
    const uint*           yolkCellsNeighbNum;
    const uint*           yolkCellsNeighbId;
    const d3*             cellRadius;
    const double          yolkMembraneRadius;
    const double          cellsYolkCmax;
    const double          cellsYolkSurfaceScaling;
    const double          cellsYolkEquilibriumDistance;
    const double          cellsYolkAttractionCoefficient;
    const double          cellsYolkRepulsionCoefficient;
    const uint*           yolkMembraneEYSL;
    const d4*             yolkMembraneTangentParams;
    const double          marginResistance;
    const uint*           yolkMembraneNextNum;
    const uint*           yolkMembraneNextId;
    d3*                   cellForces;
    uint*                 cellForcesNum;
    d3*                   yolkForces;
    uint*                 yolkForcesNum;
    uint*                 yolkMembraneEYSLupdate;
    uint*                 errorCode;

    custom_yolkcells_forces_computation(
                d3*           _yolkPosition,
                d3*           _cellPosition,
                uint*         _yolkMembraneActivated,
                uint*         _yolkCellsNeighbNum,
                uint*         _yolkCellsNeighbId,
                d3*           _cellRadius,
                double        _yolkMembraneRadius,
                double        _cellsYolkCmax,
                double        _cellsYolkSurfaceScaling,
                double        _cellsYolkEquilibriumDistance,
                double        _cellsYolkAttractionCoefficient,
                double        _cellsYolkRepulsionCoefficient,
                uint*         _yolkMembraneEYSL,
                d4*           _yolkMembraneTangentParams,
                double        _marginResistance,
                uint*         _yolkMembraneNextNum,
                uint*         _yolkMembraneNextId,
                d3*           _cellForces,
                uint*         _cellForcesNum,
                d3*           _yolkForces,
                uint*         _yolkForcesNum,
                uint*         _yolkMembraneEYSLupdate,
                uint*         _errorCode
            )
          :
            yolkPosition(_yolkPosition),
            cellPosition(_cellPosition),
            yolkMembraneActivated(_yolkMembraneActivated),
            yolkCellsNeighbNum(_yolkCellsNeighbNum),
            yolkCellsNeighbId(_yolkCellsNeighbId),
            cellRadius(_cellRadius),
            yolkMembraneRadius(_yolkMembraneRadius),
            cellsYolkCmax(_cellsYolkCmax),
            cellsYolkSurfaceScaling(_cellsYolkSurfaceScaling),
            cellsYolkEquilibriumDistance(_cellsYolkEquilibriumDistance),
            cellsYolkAttractionCoefficient(_cellsYolkAttractionCoefficient),
            cellsYolkRepulsionCoefficient(_cellsYolkRepulsionCoefficient),
            yolkMembraneEYSL(_yolkMembraneEYSL),
            yolkMembraneTangentParams(_yolkMembraneTangentParams),
            marginResistance(_marginResistance),
            yolkMembraneNextNum(_yolkMembraneNextNum),
            yolkMembraneNextId(_yolkMembraneNextId),
            cellForces(_cellForces),
            cellForcesNum(_cellForcesNum),
            yolkForces(_yolkForces),
            yolkForcesNum(_yolkForcesNum),
            yolkMembraneEYSLupdate(_yolkMembraneEYSLupdate),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){
      
      if(yolkMembraneActivated[idx] != 1 || yolkCellsNeighbNum[idx] == 0){return;}

      d3 pos1 = yolkPosition[idx];

      uint neighbCellId = yolkCellsNeighbId[idx];
      d3 relPos = cellPosition[neighbCellId] - pos1;
      double dist = length(relPos);
      double radius2 = cellRadius[neighbCellId].x;

      double surface = surface_estimation(
                                  &dist,
                                  &yolkMembraneRadius,
                                  &radius2,
                                  &cellsYolkCmax,
                                  &cellsYolkSurfaceScaling
                                  );
      
      double radsum = radius2 + yolkMembraneRadius;
      double disteq  = cellsYolkEquilibriumDistance * radsum;
      double distmax = cellsYolkCmax * radsum;

      d3 f;

      if( yolkMembraneEYSL[idx] != 1 ){

        f = attr_rep_colinear_force3( 
                            &dist,
                            &disteq,
                            &distmax,
                            &surface,
                            &cellsYolkAttractionCoefficient,
                            &cellsYolkRepulsionCoefficient
                            ) * relPos / dist;
      }
      else {
        
        d4 tp = yolkMembraneTangentParams[idx];

        d3 tangent = tp.y * (yolkPosition[(uint)(tp.x)] - pos1) + tp.w * (yolkPosition[(uint)(tp.z)] - pos1);
        tangent /= length(tangent);

        double distN = - dot(tangent, relPos);

        double nullAttr = 0;

        double newSurf = surface_estimation(
                                  &distN,
                                  &yolkMembraneRadius,
                                  &radius2,
                                  &cellsYolkCmax,
                                  &cellsYolkSurfaceScaling
                                  );

        double ftangent = attr_rep_colinear_force3( 
                          &distN,
                          &disteq,
                          &distmax,
                          &newSurf,
                          &nullAttr,
                          &cellsYolkRepulsionCoefficient
                          );

        f = - ftangent * tangent;

        if( -ftangent > marginResistance){
        // END

          if( yolkMembraneNextNum[idx] != 99999 ){ // yMNextNum == 99999 for vegetal pole

            for(uint i=0;i<yolkMembraneNextNum[idx];i++){
              uint nextCellId = yolkMembraneNextId[ 20 * idx + i ];

              if(yolkMembraneActivated[ nextCellId ] == 0){
                yolkMembraneEYSLupdate[ nextCellId ] = 1;  // 1 means new EYSL 

                //activate all previous to avoid blocking behavior
                for(uint p=0;p<yolkMembraneNextNum[NUMPARTYOLKMEMBRANE+nextCellId];p++){
                  uint idprev = yolkMembraneNextId[ 20 * NUMPARTYOLKMEMBRANE + 20 * nextCellId + p ];
                  yolkMembraneEYSLupdate[ idprev ] = 2; // 2 means old EYSL that will turn into IYSL (0)
                }
              }
            }
          }
          yolkMembraneEYSLupdate[ idx ] = 2;
        }
      }

      yolkForces[idx * NUMFORCEmax + mgAtomicAddOne(&yolkForcesNum[idx])] = f; 
      cellForces[neighbCellId * NUMFORCEmax + mgAtomicAddOne(&cellForcesNum[neighbCellId])] = - f; // * .5;


    } // end operator()
  }; // end functor custom_yolkcells_forces_computation

  struct custom_evl_forces_computation
  {
    const d3*             evlPosition;
    const d3*             evlRadius;
    const double          evlRLCoeff;
    const double          evlStiffness;
    const uint*           evlTopologicalNeighbNum;
    const uint*           evlTopologicalNeighbId;
    d3*                   evlForces;
    uint*                 evlForcesNum;
    double*               evlPressure;
    uint*                 errorCode;

    custom_evl_forces_computation(
                d3*           _evlPosition,
                d3*           _evlRadius,
                double        _evlRLCoeff,
                double        _evlStiffness,
                uint*         _evlTopologicalNeighbNum,
                uint*         _evlTopologicalNeighbId,
                d3*           _evlForces,
                uint*         _evlForcesNum,
                double*       _evlPressure,
                uint*         _errorCode
            )
          :
            evlPosition(_evlPosition),
            evlRadius(_evlRadius),
            evlRLCoeff(_evlRLCoeff),
            evlStiffness(_evlStiffness),
            evlTopologicalNeighbNum(_evlTopologicalNeighbNum),
            evlTopologicalNeighbId(_evlTopologicalNeighbId),
            evlForces(_evlForces),
            evlForcesNum(_evlForcesNum),
            evlPressure(_evlPressure),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){

      d3 pos1 = evlPosition[idx];
      double radius1 = evlRadius[idx].x;
      d3 f(0.0);
      double pressure = .0;

      uint numNeighb = evlTopologicalNeighbNum[idx];

      for(uint i = 0; i < numNeighb; i++){
        
        uint neighbIndex = idx*NUMNEIGHBTOPOmax+i;
        uint neighbCellId = evlTopologicalNeighbId[neighbIndex];
        double radius2 = evlRadius[neighbCellId].x;
        d3 relPos = evlPosition[neighbCellId] - pos1;
        double dist = length(relPos);
        relPos /= dist;

        double fevl = - evlStiffness * ( evlRLCoeff * (radius1+radius2) - dist );

        f += fevl * relPos;

        pressure += fevl;
      }
      
      if( numNeighb != 0 ){
        evlPressure[idx] = pressure / (double)numNeighb;
      }

      evlForces[idx * NUMFORCEmax + mgAtomicAddOne(&evlForcesNum[idx])] = f; 

    } // end operator()
  }; // end functor custom_evl_forces_computation

  struct custom_cellsevl_forces_computation
  {
    const d3*             evlPosition;
    const d3*             evlNormal;
    const d3*             cellPosition;
    const uint*           cellsEvlNeighbNum;
    const uint*           cellsEvlNeighbId;
    const d3*             cellRadius;
    const double          evlRadiusAB;
    const double          cellsEvlCmax;
    const double          cellsEvlSurfaceScaling;
    const double          cellsEvlEquilibriumDistance;
    const double          cellsEvlAttractionCoefficient;
    const double          cellsEvlRepulsionCoefficient;
    d3*                   cellForces;
    uint*                 cellForcesNum;
    d3*                   evlForces;
    uint*                 evlForcesNum;
    uint*                 errorCode;

    custom_cellsevl_forces_computation(
                d3*           _evlPosition,
                d3*           _evlNormal,
                d3*           _cellPosition,
                uint*         _cellsEvlNeighbNum,
                uint*         _cellsEvlNeighbId,
                d3*           _cellRadius,
                double        _evlRadiusAB,
                double        _cellsEvlCmax,
                double        _cellsEvlSurfaceScaling,
                double        _cellsEvlEquilibriumDistance,
                double        _cellsEvlAttractionCoefficient,
                double        _cellsEvlRepulsionCoefficient,
                d3*           _cellForces,
                uint*         _cellForcesNum,
                d3*           _evlForces,
                uint*         _evlForcesNum,
                uint*         _errorCode
            )
          :
            evlPosition(_evlPosition),
            evlNormal(_evlNormal),
            cellPosition(_cellPosition),
            cellsEvlNeighbNum(_cellsEvlNeighbNum),
            cellsEvlNeighbId(_cellsEvlNeighbId),
            cellRadius(_cellRadius),
            evlRadiusAB(_evlRadiusAB),
            cellsEvlCmax(_cellsEvlCmax),
            cellsEvlSurfaceScaling(_cellsEvlSurfaceScaling),
            cellsEvlEquilibriumDistance(_cellsEvlEquilibriumDistance),
            cellsEvlAttractionCoefficient(_cellsEvlAttractionCoefficient),
            cellsEvlRepulsionCoefficient(_cellsEvlRepulsionCoefficient),
            cellForces(_cellForces),
            cellForcesNum(_cellForcesNum),
            evlForces(_evlForces),
            evlForcesNum(_evlForcesNum),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){

      if(cellsEvlNeighbNum[idx] == 0){return;}

      d3 pos1 = cellPosition[idx];
      double radius1 = cellRadius[idx].x;

      uint neighbCellId = cellsEvlNeighbId[idx];

      d3 relPos = evlPosition[neighbCellId] - pos1;

      d3 normal = evlNormal[neighbCellId];

      double distN = dot(normal,relPos);

      double surface = surface_estimation(
                                  &distN,
                                  &radius1,
                                  &evlRadiusAB,
                                  &cellsEvlCmax,
                                  &cellsEvlSurfaceScaling
                                  );

      double disteq  = cellsEvlEquilibriumDistance * (radius1+evlRadiusAB);

      double f_scal = attr_rep_colinear_force2( 
                          &distN,
                          &disteq,
                          &surface,
                          &cellsEvlAttractionCoefficient,
                          &cellsEvlRepulsionCoefficient
                          );

      
      d3 f = f_scal * normal;

      cellForces[idx * NUMFORCEmax + mgAtomicAddOne(&cellForcesNum[idx])] = f; 
      evlForces[neighbCellId * NUMFORCEmax + mgAtomicAddOne(&evlForcesNum[neighbCellId])] = -f;
      
    } // end operator()
  }; // end functor custom_cellsevl_forces_computation

  struct custom_yolkmarginevl_forces_computation
  {
    const d3*             evlPosition;
    const d3*             yolkPosition;
    const uint*           yolkMarginEvlTopologicalNeighbNum;
    const uint*           yolkMarginEvlTopologicalNeighbId;
    const d3*             evlRadius;
    const double          yolkInteriorRadius;
    const double          yolkMarginEvlStiffness;
    d3*                   yolkForces;
    uint*                 yolkForcesNum;
    d3*                   evlForces;
    uint*                 evlForcesNum;
    uint*                 errorCode;

    custom_yolkmarginevl_forces_computation(
                d3*           _evlPosition,
                d3*           _yolkPosition,
                uint*         _yolkMarginEvlTopologicalNeighbNum,
                uint*         _yolkMarginEvlTopologicalNeighbId,
                d3*           _evlRadius,
                double        _yolkInteriorRadius,
                double        _yolkMarginEvlStiffness,
                d3*           _yolkForces,
                uint*         _yolkForcesNum,
                d3*           _evlForces,
                uint*         _evlForcesNum,
                uint*         _errorCode
            )
          :
            evlPosition(_evlPosition),
            yolkPosition(_yolkPosition),
            yolkMarginEvlTopologicalNeighbNum(_yolkMarginEvlTopologicalNeighbNum),
            yolkMarginEvlTopologicalNeighbId(_yolkMarginEvlTopologicalNeighbId),
            evlRadius(_evlRadius),
            yolkInteriorRadius(_yolkInteriorRadius),
            yolkMarginEvlStiffness(_yolkMarginEvlStiffness),
            yolkForces(_yolkForces),
            yolkForcesNum(_yolkForcesNum),
            evlForces(_evlForces),
            evlForcesNum(_evlForcesNum),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){

      d3 pos1 = yolkPosition[idx];

      d3 f(.0);

      for(uint i=0; i<yolkMarginEvlTopologicalNeighbNum[idx];i++){

        uint neighbIndex = idx * NUMNEIGHBTOPOmax + i;
        uint neighbCellId = yolkMarginEvlTopologicalNeighbId[neighbIndex];

        d3 relPos = evlPosition[neighbCellId] - pos1;
        double dist = length(relPos);
        relPos /= dist;

        double radius2 = evlRadius[neighbCellId].x;

        double disteq = .5 *( yolkInteriorRadius + radius2 );

        double force_scal = - yolkMarginEvlStiffness * (disteq - dist);
        
        d3 f_temp = force_scal * relPos;

        f += f_temp;

        evlForces[neighbCellId * NUMFORCEmax + mgAtomicAddOne(&evlForcesNum[neighbCellId])] = -f_temp;//  * .5;

      }

      yolkForces[idx * NUMFORCEmax + mgAtomicAddOne(&yolkForcesNum[idx])] = f ; 
      
    } // end operator()
  }; // end functor custom_yolkmarginevl_forces_computation

  struct custom_yolk_forces_integration
  {
    d3*                 yolkPosition;
    const d3*           yolkForces;
    const uint*         yolkForcesNum;
    const double        yolkInteriorRadius;
    const double        yolkMembraneRadius;
    const double        globalDamping;
    const uint          rk_loop;
    d3*                 yolk_Runge_Kutta_K;
    d3*                 yolk_Runge_Kutta_InitPos;
    const double        deltaTime;
    const d3            spatialBorderMin;
    const d3            spatialBorderMax;
    uint*               yolkMembraneEYSL;
    uint*               yolkMembraneEYSLupdate;
    uint*               yolkMembraneActivated;
    uint*               errorCode;

    custom_yolk_forces_integration(
          d3*           _yolkPosition,
          d3*           _yolkForces,
          uint*         _yolkForcesNum,
          double        _yolkInteriorRadius,
          double        _yolkMembraneRadius,
          double        _globalDamping,
          uint          _rk_loop,
          d3*           _yolk_Runge_Kutta_K,
          d3*           _yolk_Runge_Kutta_InitPos,
          double        _deltaTime,
          d3            _spatialBorderMin,
          d3            _spatialBorderMax,
          uint*         _yolkMembraneEYSL,
          uint*         _yolkMembraneEYSLupdate,
          uint*         _yolkMembraneActivated,
          uint*         _errorCode
        )
         :
            yolkPosition(_yolkPosition), 
            yolkForces(_yolkForces),
            yolkForcesNum(_yolkForcesNum),
            yolkInteriorRadius(_yolkInteriorRadius),
            yolkMembraneRadius(_yolkMembraneRadius),
            globalDamping(_globalDamping),
            rk_loop(_rk_loop),
            yolk_Runge_Kutta_K(_yolk_Runge_Kutta_K),
            yolk_Runge_Kutta_InitPos(_yolk_Runge_Kutta_InitPos),
            deltaTime(_deltaTime),
            spatialBorderMin(_spatialBorderMin),
            spatialBorderMax(_spatialBorderMax),
            yolkMembraneEYSL(_yolkMembraneEYSL),
            yolkMembraneEYSLupdate(_yolkMembraneEYSLupdate),
            yolkMembraneActivated(_yolkMembraneActivated),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){

      // Sum all forces with the Kahan summation algorithm
      d3 f(.0);
      d3 c(.0);
      uint numForces              = yolkForcesNum[idx];

      for(uint i = 0; i < numForces; i++){  
        d3 y = yolkForces[idx * NUMFORCEmax + i] - c;
        d3 t = f + y;
        c = (t - f) - y;
        f = t;
      }

      d3 pos = yolkPosition[idx]; 

      double surface;
      if(idx < NUMPARTYOLKMEMBRANE){
        surface = 4.0 * PI * yolkMembraneRadius * yolkMembraneRadius;
      }
      else{
        surface = 4.0 * PI * yolkInteriorRadius * yolkInteriorRadius;
      }
      
      double damping = 1.0 / ( globalDamping * surface );
      
      if(idx < NUMPARTYOLKMEMBRANE){
        f           *= damping;
      }
      else{
        f           *= damping;
      }
      
      //Runge Kutta integration
      double runge_Kutta_coeff[3];
      runge_Kutta_coeff[0] = .5;
      runge_Kutta_coeff[1] = .5;
      runge_Kutta_coeff[2] = 1.0;

      if(rk_loop == 0){
        yolk_Runge_Kutta_InitPos[idx] = pos;
      }

      if(rk_loop < 3){

        yolk_Runge_Kutta_K[rk_loop * NUMPARTYOLK + idx] = f * deltaTime;
        
        yolkPosition[idx] = yolk_Runge_Kutta_InitPos[idx] + runge_Kutta_coeff[rk_loop] * f;

      }
      else{
   
        pos = yolk_Runge_Kutta_InitPos[idx] +
                            (1.0 / 6.0 * 
                              ( yolk_Runge_Kutta_K[idx]
                                + 2 * yolk_Runge_Kutta_K[NUMPARTYOLK + idx]
                                + 2 * yolk_Runge_Kutta_K[2 * NUMPARTYOLK + idx]
                                + f));
                          
        //Check spatial borders
        if(pos.x < spatialBorderMin.x){pos.x = spatialBorderMin.x;}
        if(pos.y < spatialBorderMin.y){pos.y = spatialBorderMin.y;}
        if(pos.z < spatialBorderMin.z){pos.z = spatialBorderMin.z;}
        if(pos.x > spatialBorderMax.x){pos.x = spatialBorderMax.x;}
        if(pos.y > spatialBorderMax.y){pos.y = spatialBorderMax.y;}
        if(pos.z > spatialBorderMax.z){pos.z = spatialBorderMax.z;}

        yolkPosition[idx] = pos;

      }

      //propagate margin progression if required
      if(idx < NUMPARTYOLKMEMBRANE){
        if(yolkMembraneEYSLupdate[ idx ] != 0){

          if(yolkMembraneEYSLupdate[ idx ] == 1){   //new YSL
            yolkMembraneEYSL[ idx ] = 1;
          }
          else{   //old YSL
            yolkMembraneEYSL[ idx ] = 0;
          }

          yolkMembraneActivated[ idx ] = 1;
          yolkMembraneEYSLupdate[ idx ] = 0;

        }
      }

    }
  };

  struct custom_evl_forces_integration
  {
    d3*                 evlPosition;
    const d3*           evlForces;
    const uint*         evlForcesNum;
    const d3*           evlRadius;
    const double        globalDamping;
    const uint          rk_loop;
    d3*                 evl_Runge_Kutta_K;
    d3*                 evl_Runge_Kutta_InitPos;
    const double        deltaTime;
    const d3            spatialBorderMin;
    const d3            spatialBorderMax;
    const double        evlRadiusAB;

    custom_evl_forces_integration(
          d3*           _evlPosition,
          d3*           _evlForces,
          uint*         _evlForcesNum,
          d3*           _evlRadius,
          double        _globalDamping,
          uint          _rk_loop,
          d3*           _evl_Runge_Kutta_K,
          d3*           _evl_Runge_Kutta_InitPos,
          double        _deltaTime,
          d3            _spatialBorderMin,
          d3            _spatialBorderMax,
          double        _evlRadiusAB
        )
         :
            evlPosition(_evlPosition), 
            evlForces(_evlForces),
            evlForcesNum(_evlForcesNum),
            evlRadius(_evlRadius),
            globalDamping(_globalDamping),
            rk_loop(_rk_loop),
            evl_Runge_Kutta_K(_evl_Runge_Kutta_K),
            evl_Runge_Kutta_InitPos(_evl_Runge_Kutta_InitPos),
            deltaTime(_deltaTime),
            spatialBorderMin(_spatialBorderMin),
            spatialBorderMax(_spatialBorderMax),
            evlRadiusAB(_evlRadiusAB)
            {}

    __device__
    void operator()(const int& idx){

      // Sum all forces with the Kahan summation algorithm
      d3 f(.0);
      d3 c(.0);
      uint numForces = evlForcesNum[idx];

      for(uint i = 0; i < numForces; i++){  
        d3 y = evlForces[idx * NUMFORCEmax + i] - c;
        d3 t = f + y;
        c = (t - f) - y;
        f = t;
      }

      d3 pos = evlPosition[idx]; 
      d3 rad = evlRadius[idx]; 

      double surface = 4 * PI * rad.x * evlRadiusAB;

      double damping = 1.0 / ( globalDamping * surface );

      f           *= damping;

      //Runge Kutta integration
      double runge_Kutta_coeff[3];
      runge_Kutta_coeff[0] = .5;
      runge_Kutta_coeff[1] = .5;
      runge_Kutta_coeff[2] = 1.0;
      
      if(rk_loop == 0){
        evl_Runge_Kutta_InitPos[idx] = pos;
      }

      if(rk_loop < 3){

        evl_Runge_Kutta_K[rk_loop * NUMPARTEVLmax + idx] = f * deltaTime;
        
        evlPosition[idx] = evl_Runge_Kutta_InitPos[idx] + runge_Kutta_coeff[rk_loop] * f;

      }
      else{
        
        pos = evl_Runge_Kutta_InitPos[idx] +
                            (1.0 / 6.0 * 
                              ( evl_Runge_Kutta_K[idx]
                                + 2 * evl_Runge_Kutta_K[NUMPARTEVLmax + idx]
                                + 2 * evl_Runge_Kutta_K[2 * NUMPARTEVLmax + idx]
                                + f));
                          
        //Check spatial borders
        if(pos.x < spatialBorderMin.x){pos.x = spatialBorderMin.x;}
        if(pos.y < spatialBorderMin.y){pos.y = spatialBorderMin.y;}
        if(pos.z < spatialBorderMin.z){pos.z = spatialBorderMin.z;}
        if(pos.x > spatialBorderMax.x){pos.x = spatialBorderMax.x;}
        if(pos.y > spatialBorderMax.y){pos.y = spatialBorderMax.y;}
        if(pos.z > spatialBorderMax.z){pos.z = spatialBorderMax.z;}

        evlPosition[idx] = pos;

      }
      
    } 
  };

}

#endif