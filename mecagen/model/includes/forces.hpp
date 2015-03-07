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

#ifndef _FORCES_H
#define _FORCES_H

#include <cstring>
#include <stdio.h>  //printf
#include <assert.h>

namespace mg {

  /** This function calculates the attraction-repulsion force.*/
  inline
  __host__ __device__ double attr_rep_colinear_force2(
                            const double  *dist,
                            const double  *disteq,
                            const double  *surface,
                            const double  *w_adh,
                            const double  *w_rep
                            )
  {
    double deltaL = *dist - *disteq;
    double coeff;
    if( deltaL > 0 ){
      coeff = *w_adh;
    }
    else{
      coeff = *w_rep;
    }
 
    return coeff * *surface * deltaL;
  }

  /** This function calculates the attraction-repulsion force and set force to zero if distance is larger than maximum distance.*/
  inline
  __host__ __device__ double attr_rep_colinear_force3(
                            const double  *dist,
                            const double  *disteq,
                            const double  *distmax,
                            const double  *surface,
                            const double  *w_adh,
                            const double  *w_rep
                            )
  {
    double deltaL = *dist - *disteq;
    double coeff;
    if( deltaL < 0 ){
      coeff = *w_rep;
    }
    else if( *dist < *distmax){
      coeff = *w_adh;
    }
    else{
      coeff = .0;
    }
 
    return coeff * *surface * deltaL;
  }

  /** This functor calculates the force exerted between neighbor cells.*/
  struct forces_computation
  {
    const d3*       cellPosition;     
    const d3*       cellRadius;     
    const uint*     cellTopologicalNeighbId;
    const uint*     cellTopologicalNeighbNum;
    const uint*     cellNeighbIsLateral;
    const uint*     cellEpiIsPolarized;
    const uint*     cellEpiId;
    const d3*       cellAxis1;
    const d3*       cellAxisAB;
    const uint*     cellType;
    const double*   cellContactSurfaceArea;
    const double*   cellEquilibriumDistance;
    const double*   cellAttractionCoefficient;
    const double*   cellRepulsionCoefficient;
    const double*   cellPlanarRigidityCoefficient;
    const uint*     cellIntercalateWithNeighb;
    const double*   cellIntercalationIntensity;
    const uint*     cellIntercalationBipolar;
    const uint*     cellApicalConstrictionWithNeighb;
    d3*             cellForces;
    uint*           cellForcesNum;
    d3*             cellProtrusionExtForces;
    uint*           cellProtrusionExtForcesNum;
    double*         cellMechanoSensorQs;
    uint*           cellMechanoSensorQsNum;
    uint*           errorCode;
    
    forces_computation(
                d3*       _cellPosition,     
                d3*       _cellRadius,       
                uint*     _cellTopologicalNeighbId,
                uint*     _cellTopologicalNeighbNum,
                uint*     _cellNeighbIsLateral,
                uint*     _cellEpiIsPolarized,
                uint*     _cellEpiId,
                d3*       _cellAxis1,
                d3*       _cellAxisAB,
                uint*     _cellType,
                double*   _cellContactSurfaceArea,
                double*   _cellEquilibriumDistance,
                double*   _cellAttractionCoefficient,
                double*   _cellRepulsionCoefficient,
                double*   _cellPlanarRigidityCoefficient,
                uint*     _cellIntercalateWithNeighb,
                double*   _cellIntercalationIntensity,
                uint*     _cellIntercalationBipolar,
                uint*     _cellApicalConstrictionWithNeighb,
                d3*       _cellForces,
                uint*     _cellForcesNum,
                d3*       _cellProtrusionExtForces,
                uint*     _cellProtrusionExtForcesNum,
                double*   _cellMechanoSensorQs,
                uint*     _cellMechanoSensorQsNum,
                uint*     _errorCode
            )
         :
            cellPosition(_cellPosition), 
            cellRadius(_cellRadius),                      
            cellTopologicalNeighbId(_cellTopologicalNeighbId),
            cellTopologicalNeighbNum(_cellTopologicalNeighbNum),
            cellNeighbIsLateral(_cellNeighbIsLateral),
            cellEpiIsPolarized(_cellEpiIsPolarized),
            cellEpiId(_cellEpiId),
            cellAxis1(_cellAxis1),
            cellAxisAB(_cellAxisAB),
            cellType(_cellType),
            cellContactSurfaceArea(_cellContactSurfaceArea),
            cellEquilibriumDistance(_cellEquilibriumDistance),
            cellAttractionCoefficient(_cellAttractionCoefficient),
            cellRepulsionCoefficient(_cellRepulsionCoefficient),
            cellPlanarRigidityCoefficient(_cellPlanarRigidityCoefficient),
            cellIntercalateWithNeighb(_cellIntercalateWithNeighb),
            cellIntercalationIntensity(_cellIntercalationIntensity),
            cellIntercalationBipolar(_cellIntercalationBipolar),
            cellApicalConstrictionWithNeighb(_cellApicalConstrictionWithNeighb),
            cellForces(_cellForces),
            cellForcesNum(_cellForcesNum),
            cellProtrusionExtForces(_cellProtrusionExtForces),
            cellProtrusionExtForcesNum(_cellProtrusionExtForcesNum),
            cellMechanoSensorQs(_cellMechanoSensorQs),
            cellMechanoSensorQsNum(_cellMechanoSensorQsNum),
            errorCode(_errorCode)
            {}

    __device__
    void operator()(const int& idx){

      uint celltype1  = cellType[idx];
      uint cellEpiId1 = cellEpiId[idx];
      uint numNeighb  = cellTopologicalNeighbNum[idx];
      d3 pos1         = cellPosition[idx];
      d3 axis1        = cellAxis1[idx];
      d3 axisAb1      = cellAxisAB[idx];

      uint epiPolarized = (uint)(celltype1 == 2 && cellEpiIsPolarized[idx] == 1);

      d3 f(0.0), f_temp;
      double mechanoSensorQ = 0;

      for(uint i=0; i<numNeighb; i++){
        
        double mechanoSensorQ_neighb = 0;
        d3 f_neighb(.0);
        d3 f_neighb_protr_ext(.0);

        uint topoNeighbIndex    = idx * NUMNEIGHBTOPOmax + i;
        uint neighbCellId   = cellTopologicalNeighbId[topoNeighbIndex];

        d3 pos2         = cellPosition[neighbCellId];
        d3 axisAb2      = cellAxisAB[neighbCellId];
        uint celltype2  = cellType[neighbCellId];

        d3 relPos   = pos2 - pos1;
        double dist     = length(relPos);
        relPos          /= dist;

        double surface = cellContactSurfaceArea[topoNeighbIndex];
        double disteq  = cellEquilibriumDistance[topoNeighbIndex];

        /***********************************
        **** Attraction-repulsion force ****
        ***********************************/

        f_temp = attr_rep_colinear_force2( 
                          &dist,
                          &disteq,
                          &surface,
                          &(cellAttractionCoefficient[topoNeighbIndex]),
                          &(cellRepulsionCoefficient[topoNeighbIndex])
                          ) * relPos;

        mechanoSensorQ += length(f_temp);

        // the reciprocal force will be computed by the neighbor thread
        f += f_temp;

       /***********************************
        * Epithelial planar rigidity force *
        ***********************************/

        if(epiPolarized && (cellEpiId1 == cellEpiId[neighbCellId])){

          double r_lat = cellRadius[idx].x;
          double r_ab = cellRadius[idx].y;

          if( celltype2 == 2){

            d3 rigForceAxis = axisAb1 + axisAb2;
            rigForceAxis /= length(rigForceAxis);
            f_temp = cellPlanarRigidityCoefficient[topoNeighbIndex] * r_ab * r_ab * r_ab / r_lat * dot(relPos, rigForceAxis) * rigForceAxis;
            
            // this force is not reciprocal, neighbCellId is in the lateral neighborhood of idx but we do not know whether
            // idx is in the lateral neighborhood of neighbCellId
            mechanoSensorQ_neighb += length(f_temp);
            f_neighb -= f_temp;
          }
        }
        
        /***********************************
        ********* Behavioral force *********
        ***********************************/

        // Mesenchymal cells or Epithelial -> intercalation
        // cellIntercalateWithNeighb decides if some intercalation behavior occurs
        if( celltype1 >= 1 && cellIntercalateWithNeighb[topoNeighbIndex] == 1){

          // If the cell is epithelial, the force must be orthogonal to the apico-basal axis
          d3 relPosProtr = relPos;
          if(celltype1 == 2){
            relPosProtr -= dot(relPosProtr, axisAb1) * axisAb1;  
          }

          double scal = dot(relPosProtr, axis1);

          if( fabs(scal) > 0.20  && fabs(scal) < .99999 ){

            d3 forcePerp                = relPosProtr - scal * axis1;
            forcePerp                   /= length(forcePerp);

            double protrusionIntensity  = cellIntercalationIntensity[topoNeighbIndex];
            
            if(scal > 0){
              d3 forceProtrusionAxis    = forcePerp - PROTRUSION_COEFF * axis1;
              f_neighb_protr_ext = protrusionIntensity * .001 * forceProtrusionAxis / length(forceProtrusionAxis);
              mechanoSensorQ_neighb     += length(f_neighb_protr_ext);
            }
            else if( cellIntercalationBipolar[ topoNeighbIndex ] ){
              d3 forceProtrusionAxis    = forcePerp + PROTRUSION_COEFF * axis1;
              f_neighb_protr_ext = protrusionIntensity * .001 * forceProtrusionAxis / length(forceProtrusionAxis);
              mechanoSensorQ_neighb     += length(f_neighb_protr_ext);
            }
          }
        }
        
        // Epithelial cells -> apical constriction
        if( 0 && epiPolarized && cellApicalConstrictionWithNeighb[topoNeighbIndex] == 1){
          //to be added...
          // d3 apconstAxis = axisAb1 + axisAb2;
          // apconstAxis /= length(apconstAxis);     

          // f += k_ac * surface * dot(relPos, apconstAxis) * apconstAxis;
        }

        /***********************************
        **** Transfer force on neighbor ****
        ***********************************/
        f -= (f_neighb + f_neighb_protr_ext);
        mechanoSensorQ += mechanoSensorQ_neighb;

        cellForces[neighbCellId * NUMFORCEmax + mgAtomicAddOne(&cellForcesNum[neighbCellId])] = f_neighb;

        cellProtrusionExtForces[neighbCellId * NUMFORCEmax + mgAtomicAddOne(&cellProtrusionExtForcesNum[neighbCellId])] = f_neighb_protr_ext;

        cellMechanoSensorQs[neighbCellId * NUMFORCEmax + mgAtomicAddOne(&cellMechanoSensorQsNum[neighbCellId])] = mechanoSensorQ_neighb;
      }

      cellForces[idx * NUMFORCEmax + mgAtomicAddOne(&cellForcesNum[idx])] = f; 

      cellMechanoSensorQs[idx * NUMFORCEmax + mgAtomicAddOne(&cellMechanoSensorQsNum[idx])] = mechanoSensorQ;
    }
  };

  /** This functor integrates the forces and move cell positions.*/
  struct forces_integration
  {
    d3*             cellPosition;     
    const d3*       cellRadius;   
    const d3*       cellForces;
    const uint*     cellForcesNum;
    d3*             cellProtrusionExtForce;
    const d3*       cellProtrusionExtForces;
    const uint*     cellProtrusionExtForcesNum;
    const double    globalDamping;
    const uint      rk_loop;
    d3*             runge_Kutta_K;
    d3*             runge_Kutta_K_Protr_Ext;
    double*         runge_Kutta_K_Mecha_Sensor_Q;
    d3*             runge_Kutta_InitPos;
    double*         cellMechanoSensorQ;
    const double*   cellMechanoSensorQs;
    const uint*     cellMechanoSensorQsNum;
    const int       numCellsMax;
    const double    deltaTime;
    const uint*     cellType;
    const double*   cellSurface;
    const d3        spatialBorderMin;
    const d3        spatialBorderMax;

    forces_integration(
          d3*         _cellPosition,
          d3*         _cellRadius,
          d3*         _cellForces,
          uint*       _cellForcesNum,
          d3*         _cellProtrusionExtForce,
          d3*         _cellProtrusionExtForces,
          uint*       _cellProtrusionExtForcesNum,
          double      _globalDamping,
          uint        _rk_loop,
          d3*         _runge_Kutta_K,
          d3*         _runge_Kutta_K_Protr_Ext,
          double*     _runge_Kutta_K_Mecha_Sensor_Q,
          d3*         _runge_Kutta_InitPos,
          double*     _cellMechanoSensorQ,
          double*     _cellMechanoSensorQs,
          uint*       _cellMechanoSensorQsNum,
          uint        _numCellsMax,
          double      _deltaTime,
          uint*       _cellType,
          double*     _cellSurface,
          d3          _spatialBorderMin,
          d3          _spatialBorderMax
        )
         :
            cellPosition(_cellPosition), 
            cellRadius(_cellRadius),            
            cellForces(_cellForces),
            cellForcesNum(_cellForcesNum),
            cellProtrusionExtForce(_cellProtrusionExtForce),
            cellProtrusionExtForces(_cellProtrusionExtForces),
            cellProtrusionExtForcesNum(_cellProtrusionExtForcesNum),
            globalDamping(_globalDamping),
            rk_loop(_rk_loop),
            runge_Kutta_K(_runge_Kutta_K),
            runge_Kutta_K_Protr_Ext(_runge_Kutta_K_Protr_Ext),
            runge_Kutta_K_Mecha_Sensor_Q(_runge_Kutta_K_Mecha_Sensor_Q),
            runge_Kutta_InitPos(_runge_Kutta_InitPos),
            cellMechanoSensorQ(_cellMechanoSensorQ),
            cellMechanoSensorQs(_cellMechanoSensorQs),
            cellMechanoSensorQsNum(_cellMechanoSensorQsNum),
            numCellsMax(_numCellsMax),
            deltaTime(_deltaTime),
            cellType(_cellType),
            cellSurface(_cellSurface),
            spatialBorderMin(_spatialBorderMin),
            spatialBorderMax(_spatialBorderMax)
            {}

    __device__
    void operator()(const int& idx){

      // Sum all forces with the Kahan summation algorithm
      d3 f_protr_ext(.0);
      d3 c(.0);
      uint numForces              = cellForcesNum[idx];
      uint numProtrusionExtForces = cellProtrusionExtForcesNum[idx];

      for(uint i = 0; i < numProtrusionExtForces; i++){  
        d3 y = cellProtrusionExtForces[idx * NUMFORCEmax + i] - c;
        d3 t = f_protr_ext + y;
        c = (t - f_protr_ext) - y;
        f_protr_ext = t;
      }

      d3 f = f_protr_ext;
      for(uint i = 0; i < numForces; i++){ 
        d3 y = cellForces[idx * NUMFORCEmax + i] - c;
        d3 t = f + y;
        c = (t - f) - y;
        f = t;
      }
      
      // Sum all mechano sensor contributions
      double mechanoSensorQ = 0;

      for(uint i = 0; i < cellMechanoSensorQsNum[idx]; i++){  
        mechanoSensorQ += cellMechanoSensorQs[idx * NUMFORCEmax + i];
      }

      d3 pos = cellPosition[idx]; 
      double surface = cellSurface[idx];

      double damping = 1.0 / ( globalDamping * surface );
      f_protr_ext *= damping;
      f           *= damping;
     
      //Runge Kutta integration
      double runge_Kutta_coeff[3];
      runge_Kutta_coeff[0] = .5;
      runge_Kutta_coeff[1] = .5;
      runge_Kutta_coeff[2] = 1.0;

      if(rk_loop == 0){
        runge_Kutta_InitPos[idx] = pos;
      }

      if(rk_loop < 3){

        runge_Kutta_K[rk_loop * numCellsMax + idx] = f * deltaTime;
        runge_Kutta_K_Protr_Ext[rk_loop * numCellsMax + idx] = f_protr_ext * deltaTime;
        runge_Kutta_K_Mecha_Sensor_Q[rk_loop * numCellsMax + idx] = mechanoSensorQ;
        
        cellPosition[idx] = runge_Kutta_InitPos[idx] + runge_Kutta_coeff[rk_loop] * f;
        
      }
      else{
        
        // Store averaged external protrusion force exerted on idx
        // Used by setMechaState 
        cellProtrusionExtForce[idx] = 
                              (1.0 / 6.0 * 
                                  ( runge_Kutta_K_Protr_Ext[idx]
                                    + 2 * runge_Kutta_K_Protr_Ext[numCellsMax + idx]
                                    + 2 * runge_Kutta_K_Protr_Ext[2 * numCellsMax + idx]
                                    + f_protr_ext));        

        cellMechanoSensorQ[idx] = (1.0 / 6.0 * 
                                      ( runge_Kutta_K_Mecha_Sensor_Q[idx]
                                        + 2 * runge_Kutta_K_Mecha_Sensor_Q[numCellsMax + idx]
                                        + 2 * runge_Kutta_K_Mecha_Sensor_Q[2 * numCellsMax + idx]
                                        + mechanoSensorQ));

        pos = runge_Kutta_InitPos[idx] +
                            (1.0 / 6.0 * 
                              ( runge_Kutta_K[idx]
                                + 2 * runge_Kutta_K[numCellsMax + idx]
                                + 2 * runge_Kutta_K[2 * numCellsMax + idx]
                                + f));
                          
        //Check spatial borders
        if(pos.x < spatialBorderMin.x){pos.x = spatialBorderMin.x;}
        if(pos.y < spatialBorderMin.y){pos.y = spatialBorderMin.y;}
        if(pos.z < spatialBorderMin.z){pos.z = spatialBorderMin.z;}
        if(pos.x > spatialBorderMax.x){pos.x = spatialBorderMax.x;}
        if(pos.y > spatialBorderMax.y){pos.y = spatialBorderMax.y;}
        if(pos.z > spatialBorderMax.z){pos.z = spatialBorderMax.z;}

        cellPosition[idx] = pos;
        
      }
    
    }
  };

}

#endif